import pdb
from dataclasses import dataclass
from itertools import permutations, product
from random import sample, gauss
from typing import List, Dict, Union
import torch
from torch import nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartModel,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.file_utils import ModelOutput
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    try:
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    except RuntimeError as e:
        print(f"tiled_x.device={tiled_x.device}")
        print(f"tiled_y.device={tiled_y.device}")
        raise e
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def sample_z(z_dim: int, iterations: int = 100) -> torch.Tensor:
    x = torch.randn(iterations, z_dim)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def sample_node_subset(max_num_nodes: int, avg_num_nodes: float = 3.0, sigma: float = 0) -> List[int]:
    num_nodes = min(max_num_nodes, round(gauss(avg_num_nodes, sigma)))
    node_subset = sample(list(range(max_num_nodes)), k=num_nodes)
    return node_subset


def sample_edges(max_num_nodes, node_subset, avg_num_interactions, sigma: float = 0):
    all_pairwise_interactions = list(product(node_subset, node_subset))
    adj_matrix = torch.zeros(max_num_nodes, max_num_nodes)
    max_num_interactions = len(node_subset) ** 2
    num_interactions = min(round(gauss(mu=avg_num_interactions, sigma=sigma)), max_num_interactions)
    pairwise_interactions = sample(all_pairwise_interactions, k=num_interactions)
    adj_matrix[list(zip(*pairwise_interactions))] = 1.0
    return adj_matrix


def sample_node_labels(max_num_nodes, node_subset, node_features=10):
    v = torch.zeros(max_num_nodes, node_features)
    p = torch.full_like(v[node_subset], 0.5)
    sample = torch.bernoulli(p)
    v[node_subset] = sample
    return v


def sample_graph(max_num_nodes, num_entities, num_interactions, num_entity_features, iterations=10):
    # TODO: plot distribution of mmd distance bewteen random pairs
    edges = []
    node_embeddings = []
    for i in range(iterations):
        node_subset = sample_node_subset(max_num_nodes, avg_num_nodes=num_entities)
        adj_matrix = sample_edges(max_num_nodes, node_subset, num_interactions)
        edges.append(adj_matrix.unsqueeze(0))
        labeled_nodes = sample_node_labels(max_num_nodes, node_subset, num_entity_features)
        node_embeddings.append(labeled_nodes.unsqueeze(0))
    edges = torch.cat(edges, 0)
    node_embeddings = torch.cat(node_embeddings, 0)
    return edges, node_embeddings


def compute_mmd_loss(z: torch.Tensor, iterations: int) -> torch.Tensor:
    # https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
    z_dim = z.size(-1)
    z_samples = sample_z(z_dim, iterations)
    z_loss = mmd(z_samples, z)
    return z_loss


def compute_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # https://github.com/timbmg/Sentence-VAE/blob/master/train.py
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    return kl.sum()


def monte_carlo_kl_divergence(self, z, mu, std):
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # this has the advantage that one can choose the distribution, and in particular, mu and std
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))  # N(0, 1) could be somethign else
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl


def compute_loss_on_twins(z: List[torch.Tensor]) -> torch.Tensor:
    assert len(z) == 2, "for the moment, this works only on twin pairs, not for higher order"
    assert z[0].size() == z[1].size(), "z dims have to be equal for square cross correl matrix"
    # z = [t.cpu() for t in z]
    batch_size, z_dim = z[0].size()
    c = (z[0].T @ z[1]) / batch_size
    diag = c.diagonal()
    off_diag = c - torch.diag_embed(diag)
    loss_diag = (diag - 1) ** 2
    loss_off_diag = off_diag ** 2
    loss_diag = loss_diag.sum() / z_dim  # num elements of diag scales as n
    loss_off_diag = loss_off_diag.sum() / ((z_dim ** 2) - z_dim)  # num elements off_diag roughly scales as n^2 - n
    # if torch.cuda.is_available():
    #     loss_diag = loss_diag.cuda()
    #     loss_off_diag = loss_off_diag.cuda()
    #     c = c.cuda()
    return loss_diag, loss_off_diag, c


class LatentConfig(BartConfig):
    # inherited from BartConfig:
    #
    # vocab_size=50265,
    # max_position_embeddings=1024,
    # encoder_layers=12,
    # encoder_ffn_dim=4096,
    # encoder_attention_heads=16,
    # decoder_layers=12,
    # decoder_ffn_dim=4096,
    # decoder_attention_heads=16,
    # encoder_layerdrop=0.0,
    # decoder_layerdrop=0.0,
    # activation_function="gelu",
    # d_model=1024,
    # dropout=0.1,
    # attention_dropout=0.0,
    # activation_dropout=0.0,
    # init_std=0.02,
    # classifier_dropout=0.0,
    # scale_embedding=False,
    # use_cache=True,
    # num_labels=3,
    # pad_token_id=1,
    # bos_token_id=0,
    # eos_token_id=2,
    # is_encoder_decoder=True,
    # decoder_start_token_id=2,
    # forced_eos_token_id=2,

    keys_to_ignore_at_inference = ['supp_data']

    def __init__(
        self,
        freeze_pretrained: str = 'both',
        hidden_features: int = 100,
        z_dim: int = 128,
        sampling_iterations: int = 100,
        seq_length: int = 512,
        latent_var_loss: str = 'mmd',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_pretrained = freeze_pretrained
        self.hidden_features = hidden_features
        self.z_dim = z_dim
        self.sampling_iterations = sampling_iterations
        self.seq_length = seq_length
        self.latent_var_loss = latent_var_loss


class GraphLatentConfig(LatentConfig):
    def __init__(
        self,
        mlp_num_layers: int = 1,
        num_nodes: int = 5,
        num_entity_features: int = 32,
        sample_num_interactions: int = 20,
        sample_num_entities: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlp_num_layers = mlp_num_layers
        self.num_nodes = num_nodes
        self.num_entity_features = num_entity_features
        self.sample_num_interactions = sample_num_interactions
        self.sample_num_entities = sample_num_entities
        self.z_dim = (self.num_nodes ** 2) + (self.num_nodes * self. num_entity_features)  # concat vectorized adj matrix and entity embeddings, overrides z_dim
        self.alpha = alpha
        self.beta = beta


class VAEConfig(LatentConfig):
    def __init__(self, residuals: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.residuals = residuals


class VAEConfigLM(VAEConfig):

    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # weights of lm loss when composed with loss on latent var z


class VAEConfigForTokenClassification(LatentConfig):

    def __init__(self, classifier_dropout: float = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_dropout = classifier_dropout


class TwinConfig(BartConfig):

    def __init__(self, lambd_a: float = None, mu: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd_a = lambd_a  # not a typo; weight on off diagonal terms of twin loss
        self.mu = mu  # weight twin z loss vs the other losses


class GraphVAEConfigLM(VAEConfigLM):
    def __init__(
        self,
        mlp_num_layers: int = 1,
        num_nodes: int = 5,
        num_entity_features: int = 32,
        sample_num_interactions: int = 20,
        sample_num_entities: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlp_num_layers = mlp_num_layers
        self.num_nodes = num_nodes
        self.num_entity_features = num_entity_features
        self.sample_num_interactions = sample_num_interactions
        self.sample_num_entities = sample_num_entities
        self.z_dim = (self.num_nodes ** 2) + (self.num_nodes * self. num_entity_features)  # concat vectorized adj matrix and entity embeddings, overrides z_dim
        self.alpha = alpha
        self.beta = beta

@dataclass
class LatentEncoderOutput(ModelOutput):
    loss: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    z: torch.Tensor = None
    representation: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class LatentDecoderOutput(ModelOutput):
    loss: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    z: torch.Tensor = None
    representation: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class VAEOutput(ModelOutput):
    loss: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    z: torch.Tensor = None
    representation: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class VAELMOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    z: torch.Tensor = None
    representation: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class TwinOutput(ModelOutput):
    loss: torch.Tensor = None
    last_hidden_state: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class TwinLMOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    supp_data: Dict[str, torch.Tensor] = None


class LatentEncoder(nn.Module):

    def __init__(
        self,
        pretrained,
        config: LatentConfig
    ):
        super().__init__()
        self.config = config
        self.freeze_pretrained = self.config.freeze_pretrained
        self.model = pretrained
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'encoder']:
            for param in self.model.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained == '':
            pass
        else:
            raise ValueError(f"not sure what to freeze or not with freeze_pretrained={self.freeze_pretrained}")

        self.d_encoder = self.model.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.model.config.pad_token_id
        # latent vars
        self.hidden_features = self.config.hidden_features
        self.sampling_iterations = self.config.sampling_iterations
        self.z_dim = self.config.z_dim
        self.latent_var_loss = self.config.latent_var_loss
        # own layers
        self.act_fct = nn.GELU()
        self.vae_dropout = nn.Dropout(p=config.dropout)
        self.fc_compress = nn.Linear(self.d_encoder, self.hidden_features)
        self.norm_compress = nn.LayerNorm(self.hidden_features, elementwise_affine=False)
        if self.latent_var_loss == "mmd" or self.latent_var_loss is None:   # infoVAE
            self.fc_z_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
        elif self.latent_var_loss == "kl":   # classical VAE
            self.fc_z_mean = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
            self.fc_z_logvar = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")
        self.norm_z = nn.LayerNorm(self.z_dim, elementwise_affine=False)

    def forward(self, input_ids=None, labels=None, **kwargs) -> LatentEncoderOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.model(input_ids=input_ids, **kwargs)
        x = encoder_outputs.last_hidden_state  # -> B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"observed seq length {length} mismatches with config.seq_length {self.seq_length} with input_ids.size()={input_ids.size()}"
        # compress
        y = x  # keep x for later as residual
        y = self.vae_dropout(y)
        y = self.fc_compress(y)  # -> B x L x H (example: 32 example x 256 token x 256 hidden features)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 65_536)
        # latent var
        y = self.vae_dropout(y)
        if self.latent_var_loss == "mmd" or self.latent_var_loss is None:
            z = self.fc_z_1(y)  # -> B x Z  (example: 32 example x 128 dimensional latent var)
            z = self.norm_z(z)
            representation = z
        elif self.latent_var_loss == "kl":
            z_mean = self.fc_z_mean(y)  # -> B x Z
            z_logvar = self.fc_z_logvar(y)  # -> B x Z
            z_std = torch.exp(0.5 * z_logvar)
            z = sample_z(self.z_dim, batch_size)
            z = z + z_mean + z_std
            representation = self.norm_z(z_mean)  # for twin cross correlation: take latent before sampling
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        if self.latent_var_loss == "mmd":
            loss = compute_mmd_loss(z, self.sampling_iterations)
        elif self.latent_var_loss == "kl":
            loss = compute_kl_loss(z_mean, z_logvar)
            # loss = float(1/(1 + exp(-k * (step - x0))))  #  would need access to training_step, modify Trainer class
        elif self.latent_var_loss is None:
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        return LatentEncoderOutput(
            loss=loss,
            last_hidden_state=x,
            z=z,
            representation=representation,
            supp_data={"loss_z": loss}
        )


class LatentDecoder(nn.Module):

    def __init__(
        self,
        pretrained,
        config: LatentConfig
    ):
        super().__init__()
        self.config = config
        self.freeze_pretrained = self.config.freeze_pretrained
        self.model = pretrained
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'decoder']:
            for param in self.model.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained == '':
            pass
        else:
            raise ValueError(f"not sure what to freeze or not with freeze_pretrained={self.freeze_pretrained}")

        self.d_decoder = self.model.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.model.config.pad_token_id
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.residuals = self.config.residuals
        # latent vars
        self.hidden_features = self.config.hidden_features
        self.z_dim = self.config.z_dim
        # own layers
        self.act_fct = nn.GELU()
        self.vae_dropout = nn.Dropout(p=config.dropout)
        self.fc_z_2 = nn.Linear(self.z_dim, self.seq_length * self.hidden_features)
        self.norm_decompress = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)
        self.fc_decompress = nn.Linear(self.hidden_features, self.d_decoder)

    def forward(self, input_ids=None, encoder_outputs: LatentEncoderOutput = None, **kwargs) -> LatentDecoderOutput:
        x = encoder_outputs.last_hidden_state
        z = encoder_outputs.z
        batch_size, z_dim = z.size()
        # decompress
        y = self.fc_z_2(z)  # -> B x (L * H)
        y = self.norm_decompress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        y = self.fc_decompress(y)  # -> B x L x H_dec
        if self.residuals:
            y = x + y  # resnet style
        # decoder
        decoder_input_ids = shift_tokens_right(
            input_ids,
            self.pad_token_id,
            self.decoder_start_token_id
        )
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.model(
            input_ids=decoder_input_ids,
            encoder_hidden_states=y,
            **kwargs
        )

        last_hidden_state = decoder_outputs.last_hidden_state

        loss = torch.tensor(0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        return LatentDecoderOutput(
            loss=loss,
            last_hidden_state=last_hidden_state
        )


class VAE(nn.Module):

    def __init__(
        self,
        encoder: LatentEncoder,
        decoder: LatentDecoder,
        config: LatentConfig
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = self.encoder.z_dim

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        encoder_outputs: LatentEncoderOutput = self.encoder(input_ids=input_ids, **kwargs)
        decoder_outputs = self.decoder(input_ids=input_ids, encoder_outputs=encoder_outputs)

        last_hidden_state = decoder_outputs.last_hidden_state

        loss_z = encoder_outputs.loss
        loss_decoder = decoder_outputs.loss
        loss = loss_z + loss_decoder

        return VAEOutput(
            loss=loss,
            last_hidden_state=last_hidden_state,
            z=encoder_outputs.z,
            representation=encoder_outputs.representation,
            supp_data={"loss_z": loss}
        )


class VAEForLM(VAE):

    def __init__(self, pretrained: BartForConditionalGeneration, config: VAEConfigLM, **kwargs):
        model = self._build_model(pretrained, config)
        super().__init__(
            model.encoder,
            model.decoder,
            config,
            **kwargs
        )
        self.gamma = self.config.gamma
        self.model = model
        self.lm_head = nn.Linear(pretrained.config.d_model, pretrained.shared.num_embeddings, bias=False)

    @staticmethod
    def _build_model(pretrained, config):
        return VAE(
            LatentEncoder(pretrained.get_encoder(), config),
            LatentDecoder(pretrained.get_decoder(), config),
            config
        )

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        outputs = self.model(input_ids, **kwargs)
        # outputs = super().forward(input_ids, labels, **kwargs)

        # trainable language model head
        logits = self.lm_head(outputs.last_hidden_state)
        supp_data = outputs.supp_data

        # calculate composite loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_lm = self.gamma * loss_fct(logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            loss_z = outputs.loss  # loss on latent var
            loss = loss_lm + loss_z  # combine with language modelling loss
            supp_data['loss_lm'] = loss_lm  # keep track for plotting in TensorBoard
        else:
            loss = None
            supp_data['loss_lm'] = loss_lm = None
        return VAELMOutput(
            loss=loss,
            logits=logits,
            z=outputs.z,
            representation=outputs.representation,
            supp_data=supp_data
        )


class Twin(nn.Module):

    def __init__(
        self,
        models: Union[LatentEncoder, List[LatentEncoder]],
        config: TwinConfig
    ):
        super().__init__()
        self.models = nn.ModuleList(models) if isinstance(models, list) else nn.ModuleList([models])
        self.config = config
        self.z_dims = [m.z_dim for m in self.models]
        self.mu = self.config.mu
        self.lambd_a = self.config.lambd_a

    def forward(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.twin_prediction(input_ids, labels, attention_mask, **kwargs)

        loss, loss_twin_z, loss_diag, loss_off_diag, cross_correl = self.all_losses(outputs)

        representations = [out.representation for out in outputs]  # List[B X Z]

        supp_data = {
                "loss_diag": loss_diag,
                "loss_off_diag": loss_off_diag,
                "loss_twin_z": loss_twin_z,
                "img_correl": cross_correl.unsqueeze(0),
            }

        supp_data = self.update_supp_data(supp_data, outputs)

        return TwinOutput(
            loss=loss,
            last_hidden_state=[out.last_hidden_state for out in outputs],
            representations=representations,
            supp_data=supp_data
        )

    def twin_prediction(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ) -> Union[List[LatentEncoderOutput], List[VAELMOutput]]:

        # note: there are no labels for encoder-only Twin models
        if len(self.models) == 1:  # single model for all twin examples
            outputs = [
                self.models[0](input_ids=input_ids[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
        else:  # one model trained for each type fo twin example
            outputs = [
                self.models[i](input_ids=input_ids[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
        return outputs

    def all_losses(self, outputs):
        loss_diag, loss_off_diag, cross_correl = compute_loss_on_twins([out.representation for out in outputs])
        losses = torch.stack([out.loss for out in outputs])
        losses = losses.sum()
        loss_twin_z = self.mu * (loss_diag + self.lambd_a * loss_off_diag)
        loss = losses + loss_twin_z
        return loss, loss_twin_z, loss_diag, loss_off_diag, cross_correl

    @staticmethod
    def update_supp_data(supp_data, outputs):
        for i, out in enumerate(outputs):
            for k, v in out.supp_data.items():
                supp_data[f"{k}_{i}"] = v
        return supp_data


class TwinLM(Twin):

    def __init__(
        self,
        models: Union[VAEForLM, List[VAEForLM]],
        config: TwinConfig
    ):
        super().__init__(models, config)

    def forward(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.twin_prediction(input_ids, labels, attention_mask, **kwargs)

        loss, loss_twin_z, loss_diag, loss_off_diag, cross_correl = self.all_losses(outputs)

        representations = [out.representation for out in outputs]  # List[B X Z]

        supp_data = {
                "loss_diag": loss_diag,
                "loss_off_diag": loss_off_diag,
                "loss_twin_z": loss_twin_z,
                "img_correl": cross_correl.unsqueeze(0),
                # "embeddings": torch.cat(representations, 0)  # not so easy...since trainer averages across GPUs
            }

        supp_data = self.update_supp_data(supp_data, outputs)

        # the only difference with parent class: get logits from language model head
        logits = [out.logits for out in outputs]

        return TwinLMOutput(
            loss=loss,
            logits=logits,
            representations=representations,
            supp_data=supp_data
        )

    def twin_prediction(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ) -> Union[List[LatentEncoderOutput], List[VAELMOutput]]:

        # note: labels are required for language model Twin models
        if len(self.models) == 1:  # single model for all twin examples
            outputs = [
                self.models[0](input_ids=input_ids[i], labels=labels[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
        else:  # one model trained for each type fo twin example
            outputs = [
                self.models[i](input_ids=input_ids[i], labels=labels[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
        return outputs


class MLP(nn.Module):
    def __init__(self, num_layer: int, N: int):
        super().__init__()
        self.N = N
        m = []
        for i in range(num_layer):
            m.append(nn.Linear(self.N, self.N))
            m.append(nn.GELU())
            m.append(nn.LayerNorm(self.N))
        self.model = nn.Sequential(*m)

    def forward(self, x):
        y = x.view(-1, self.N)
        y = self.model(y)
        y = y.view_as(x)
        return y


def permute_columns_rows(adj: torch.Tensor, entities: torch.Tensor) -> List[torch.Tensor]:
    assert adj.size(-1) == adj.size(-2), f"Expecting square matrices but received {str(adj.size())}."  # check it is a square matrix in its last dimensions
    assert adj.size(-1) == entities.size(-1),  f"Expecting same number of nodes and entties, got {adj.size()} and {entities.size()}"
    d = adj.size(-1)
    indices_permutations = list(permutations(range(d)))
    perm_adj = [adj[:, :, indices][:, indices, :] for indices in indices_permutations]
    perm_entities = [entities[:, :, indices] for indices in indices_permutations]
    return perm_adj, perm_entities


class GraphEncoder(nn.Module):

    def __init__(
        self,
        pretrained,
        config: GraphLatentConfig
    ):
        super().__init__()
        self.config = config
        self.num_nodes = self.config.num_nodes
        self.num_entity_features = self.config.num_entity_features

        self.freeze_pretrained = self.config.freeze_pretrained
        self.model = pretrained
        self.mlp_num_layers = self.config.mlp_num_layers
        self.mlp_graph_sigma = MLP(self.mlp_num_layers, self.num_nodes ** 2)
        self.mlp_graph_rho = MLP(self.mlp_num_layers, self.num_nodes ** 2)
        self.mlp_entity_sigma = MLP(self.mlp_num_layers, self.num_nodes * self.num_entity_features)
        self.mlp_entity_rho = MLP(self.mlp_num_layers, self.num_nodes * self.num_entity_features)
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'encoder']:
            for param in self.model.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained == '':
            pass
        else:
            raise ValueError(f"not sure what to freeze or not. Received freeze_pretrained={self.freeze_pretrained}")

        self.d_encoder = self.model.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.model.config.pad_token_id
        # adj matrix
        # latent vars
        self.hidden_features = self.config.hidden_features
        self.sampling_iterations = self.config.sampling_iterations
        self.latent_var_loss = self.config.latent_var_loss
        self.z_dim = self.config.z_dim
        assert self.z_dim == (self.num_nodes * self.num_entity_features) + (self.num_nodes ** 2), f"{self.z_dim} <> {(self.num_nodes * self.num_entity_features) + (self.num_nodes ** 2)}"
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        # own layers
        self.act_fct = nn.GELU()
        self.vae_dropout = nn.Dropout(p=config.dropout)
        self.fc_compress = nn.Linear(self.d_encoder, self.hidden_features)
        self.norm_compress = nn.LayerNorm(self.hidden_features, elementwise_affine=False)
        # latent adjascency matrix
        self.to_adj_matrix = nn.Linear(self.seq_length * self.hidden_features, self.num_nodes ** 2)
        self.norm_adj = nn.LayerNorm(self.num_nodes ** 2, elementwise_affine=False)
        # latent entity embeeddings
        self.to_entity_embed = nn.Linear(self.seq_length * self.hidden_features, self.num_nodes * self.num_entity_features)
        self.norm_entities = nn.LayerNorm(self.num_nodes * self.num_entity_features, elementwise_affine=False)
        # sampling param
        self.sampling_iterations = self.config.sampling_iterations
        self.sample_num_interactions = self.config.sample_num_interactions
        self.sample_num_entities = self.config.sample_num_entities

    def forward(self, input_ids=None, labels=None, **kwargs) -> LatentEncoderOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.model(input_ids=input_ids, **kwargs)
        x = encoder_outputs.last_hidden_state  # -> B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"observed seq length {length} mismatches with config.seq_length {self.seq_length} with input_ids.size()={input_ids.size()}"

        # compress
        y = x  # keep x for later as residual
        y = self.vae_dropout(y)
        y = self.fc_compress(y)  # -> B x D x D (example: 32 example x 256 token x 256 hidden features)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))

        # adj matrix
        adj = self.vae_dropout(y)
        adj = self.to_adj_matrix(adj)
        adj = self.norm_adj(adj)
        adj = self.act_fct(adj)
        adj_matrix_representation = adj
        adj = adj.view(-1, self.num_nodes, self.num_nodes)

        # entities
        entities = self.vae_dropout(y)
        entities = self.to_entity_embed(entities)
        entities = self.norm_entities(entities)
        entities = self.act_fct(entities)
        entities_representation = entities
        entities = entities.view(-1, self.num_entity_features, self.num_nodes)

        # permutation sets
        z_graph, z_entities = self.to_permuation_independent_set(adj, entities)

        if self.latent_var_loss == "mmd":
            loss, supp_data = self.compute_loss_on_latent_var(z_graph, z_entities)
        elif self.latent_var_loss is None:
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        representation = [adj_matrix_representation, entities_representation]
        z_graph = z_graph.view(-1, self.num_nodes * self.num_nodes)
        z_entities = z_entities.view(-1, self.num_entity_features * self.num_nodes)
        z = torch.cat([z_graph, z_entities], -1)

        return LatentEncoderOutput(
            loss=loss,
            last_hidden_state=x,
            z=z,
            representation=representation,
            supp_data={
                "loss_z": loss,
                "adj_distro_loss": supp_data["adj_distro_loss"],
                "nodes_distro_loss": supp_data["nodes_distro_loss"],
            }
        )

    def to_permutation_independent_set(self, adj, entities):
        # should this be with torch.no_grad() when using in loss mmd?
        permuted_adj, permuted_entities = permute_columns_rows(adj, entities)
        z_graph = self.mlp_graph_rho(
            sum([
                self.mlp_graph_sigma(x)
                for x in permuted_adj
            ])
        )
        z_entities = self.mlp_entity_rho(
            sum([
                self.mlp_entity_sigma(x)
                for x in permuted_entities
            ])
        )
        return z_graph, z_entities

    def compute_loss_on_latent_var(self, z_graph, z_entities):
        with torch.no_grad():
            edge_sample, entity_sample = sample_graph(
                self.num_nodes,
                self.sample_num_entities,
                self.sample_num_interactions,
                self.num_entity_features,
                self.sampling_iterations
            )
            z_graph_sample, z_entities_sample = self.to_permutation_independent_set(edge_sample, entity_sample)

        adj_matrix_distro_loss = self.alpha * mmd(
            z_graph_sample.view(self.sampling_iterations, self.num_nodes ** 2),
            z_graph.view(-1, self.num_nodes ** 2)
        )
        entity_distro_loss = self.beta * mmd(
            z_entities_sample.view(self.sampling_iterations, self.num_nodes * self.num_entity_features),
            z_entities.view(-1,  self.num_nodes * self.num_entity_features)
        )

        diag = z_graph.diagonal()
        loss_diag = diag ** 2
        loss_diag = loss_diag.sum() / (self.num_nodes ** 2)  # num elements of diag scales as n

        L_adj_sparse = z_graph.abs().mean()
        L_node_sparse = z_entities.abs().mean()

        # # https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
        # # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
        # # naive (me...):
        # batch_size = z_1.size(0)
        # d = self.num_nodes  # cosmetic
        # W = z_1.view(batch_size, d, d)
        # I = torch.eye(d).unsqueeze(0).expand(batch_size, d, d)
        # if torch.cuda.is_available():
        #     W = W.cuda()
        #     I = I.cuda()
        # mat_power_d = torch.matrix_power(I + (W * W ) / d, d)  # based on below Yu et al
        # trace = mat_power_d.diagonal(dim1=-1, dim2=-2).sum(-1)
        # L_dag = self.gamma * (trace - d).mean()

        # Zheng et al 2018 DAG with NOTEARS
        # implementation in https://github.com/xunzheng/notears/blob/master/notears/linear.py
        # Section 3.2 The general case: Weighted adjacency matrices
        # E = torch.matrix_exp(W * W)  # (Zheng et al. 2018)
        # h = E.diagonal(dim1=-1, dim2=-2).sum(-1) - d
        # L_dag = h.mean()
        # in NOTEARS github code:
        # A different formulation, slightly faster at the cost odf numerical stability
        # (Yu et al. 2019) DAG-GNN: DAG Structure Learning with Graph Neural Networks
        # M = np.eye(d) + W * W / d
        # E = np.linalg.matrix_power(M, d - 1)  # why d -1 with matrix power and then element wise E.T * M below?
        # h = (E.T * M).sum() - d

        loss = adj_matrix_distro_loss + entity_distro_loss  + L_adj_sparse + L_node_sparse + loss_diag
        supp_data = {
            "adj_distro_loss": adj_matrix_distro_loss,
            "nodes_distro_loss": entity_distro_loss,
            "L_adj_sparse": L_adj_sparse,
            # "L_dag": L_dag,
            "L_node_sparse": L_node_sparse,
            "loss_diag": loss_diag,
        }
        return loss, supp_data


class GraphVAEForLM(VAEForLM):

    def __init__(self, pretrained: BartForConditionalGeneration, config: GraphVAEConfigLM, **kwargs):
        super().__init__(pretrained, config, **kwargs)


    @staticmethod
    def _build_model(pretrained, config):
        return VAE(
            GraphEncoder(pretrained.get_encoder(), config),
            LatentDecoder(pretrained.get_decoder(), config),
            config
        )
