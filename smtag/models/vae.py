from dataclasses import dataclass
from itertools import permutations, product
from random import sample, gauss, random
from typing import List, Dict, Tuple, Union, Any
import torch
from torch import nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration, BartPretrainedModel,
    BartModel,
    PreTrainedModel,
)
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartEncoder, BartDecoder, BartAttention,
    _expand_mask
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput, Seq2SeqLMOutput
)
from transformers.file_utils import ModelOutput
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from .dir_attention import FlippableBartEncoder


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
    x = torch.randn(iterations, z_dim)  # random numbers from a normal distribution with mean 0 and variance 1
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
    v = torch.zeros(node_features, max_num_nodes)
    p = torch.full_like(v[:, node_subset], 0.5)
    sample = torch.bernoulli(p)
    v[:, node_subset] = sample
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
    return kl.sum()  # or kl mean()??


def monte_carlo_kl_divergence(z, mu, std):
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
    return kl.mean()  # or kl.sum()?


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


def flip(t: torch.Tensor) -> torch.Tensor:
    return t.flip(1) if t is not None else None


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
        flip_proba: float = 0,
        flipped: torch.Tensor = torch.tensor(False),
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
        self.flip_proba = flip_proba
        self.flipped = flipped


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


class TwinConfig(LatentConfig):

    def __init__(self, lambd: float = None, mu: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = lambd  # not a typo; weight on off diagonal terms of twin loss
        self.mu = mu  # weight twin z loss vs the other losses


class TwinLMConfig(TwinConfig):

    def __init__(self, gamma: float = None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma


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


# class FlipBartConfig extending BartConfig to specify number of flip layers
class FlipBartConfig(BartConfig):
    def __init__(self, num_flip_layers=0, freeze_pretrained: str='both', **kwargs):
        super().__init__(**kwargs)
        self.num_flip_layers = num_flip_layers
        self.freeze_pretrained = freeze_pretrained


@dataclass
class LatentEncoderOutput(BaseModelOutput):
    # redefine them to preserve order
    last_hidden_state: torch.FloatTensor = None
    hidden_states = None
    attentions = None
    loss: torch.Tensor = None
    latent_variable: torch.Tensor = None
    representation: torch.Tensor = None
    hidden_before_latent: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None
    # inherited
    # last_hidden_state: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LatentCGraphEncoderOutput(LatentEncoderOutput):
    flipped: torch.Tensor = None

@dataclass
class VAEOutput(Seq2SeqModelOutput):
    loss: torch.Tensor = None
    latent_variable: torch.Tensor = None
    representation: torch.Tensor = None
    hidden_before_latent: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class VAELMOutput(Seq2SeqLMOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    latent_variable: torch.Tensor = None
    representation: torch.Tensor = None
    hidden_before_latent: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


@dataclass
class CGVAELMOutput(VAELMOutput):
    flipped: bool = None


@dataclass
class TwinOutput(LatentEncoderOutput):
    loss: torch.Tensor = None
    last_hidden_state: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    hidden_before_latent: List[torch.Tensor] = None
    latent_variable: List[torch.Tensor] = None
    supp_data: Dict[str, torch.Tensor] = None
    last_hidden_state: List[torch.FloatTensor] = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TwinLMOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    hidden_before_latent: torch.Tensor = None
    supp_data: Dict[str, torch.Tensor] = None


class LatentEncoder(BartEncoder):

    def __init__(
        self,
        pretrained_encoder,
        config: LatentConfig
    ):
        super().__init__(config)
        self.config = config
        self.freeze_pretrained = self.config.freeze_pretrained
        self.model = pretrained_encoder
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'encoder']:
            for param in self.model.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained in ['', 'decoder']:
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
        elif self.latent_var_loss == "kl" or "kl-mc":   # classical VAE
            self.fc_z_mean = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
            self.fc_z_logvar = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")
        self.norm_z = nn.LayerNorm(self.z_dim, elementwise_affine=False)

    def forward(
        self,
        input_ids=None,
        **kwargs,
        # attention_mask=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
    ) -> LatentEncoderOutput:
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
        hidden_before_latent = y  # for visualization
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 65_536)
        # latent var
        y = self.vae_dropout(y)
        if self.latent_var_loss == "mmd":
            z = self.fc_z_1(y)  # -> B x Z  (example: 32 example x 128 dimensional latent var)
            z = self.norm_z(z)
            loss = compute_mmd_loss(z, self.sampling_iterations)
            representation = z
        elif self.latent_var_loss == "kl":
            z_mean = self.fc_z_mean(y)  # -> B x Z
            z_logvar = self.fc_z_logvar(y)  # -> B x Z
            z_std = torch.exp(0.5 * z_logvar)
            # z = sample_z(self.z_dim, batch_size)
            # z = z + z_mean + z_std
            q = torch.distributions.Normal(z_mean, z_std)
            z = q.rsample()
            representation = self.norm_z(z_mean)  # for twin cross correlation: take latent before sampling
            loss = compute_kl_loss(z_mean, z_logvar)
        elif self.latent_var_loss == "kl-mc":
            z_mean = self.fc_z_mean(y)  # -> B x Z
            z_logvar = self.fc_z_logvar(y)  # -> B x Z
            z_std = torch.exp(0.5 * z_logvar / 2)
            q = torch.distributions.Normal(z_mean, z_std)
            z = q.rsample()
            representation = self.norm_z(z_mean)  # for twin cross correlation: take latent before sampling
            loss = monte_carlo_kl_divergence(z, z_mean, z_std)
        elif self.latent_var_loss is None:
            z = self.fc_z_1(y)  # -> B x Z  (example: 32 example x 128 dimensional latent var)
            z = self.norm_z(z)
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
            representation = z
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        supp_data = {
            "loss_z": loss,
        }

        return LatentEncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            loss=loss,
            latent_variable=z,
            hidden_before_latent=hidden_before_latent,
            representation=representation,
            supp_data=supp_data,
        )


class LatentDecoder(BartDecoder):

    def __init__(
        self,
        pretrained_decoder,
        config: LatentConfig
    ):
        super().__init__(config)
        self.config = config
        self.freeze_pretrained = self.config.freeze_pretrained
        self.model = pretrained_decoder
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'decoder']:
            for param in self.model.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained in ['', 'encoder']:
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
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        latent_variable=None,  # hallmark of VAE
        attention_mask=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        z = latent_variable
        batch_size, z_dim = z.size()
        # decompress
        y = self.fc_z_2(z)  # -> B x (L * H)
        y = self.norm_decompress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        y = self.fc_decompress(y)  # -> B x L x H_dec
        if self.residuals:
            y = encoder_hidden_states + y  # resnet style
        # decoder
        decoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=y, # TRY FOR TESTING: encoder_hidden_states
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs
        # BaseModelOutputWithPastAndCrossAttentions
        # return LatentDecoderOutput(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     past_key_values=decoder_outputs.past_key_values,
        #     hidden_states=decoder_outputs.hidden_states,
        #     attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        # )


class VAE(BartModel):

    def __init__(
        self,
        config: LatentConfig,
        pretrained_encoder: LatentEncoder,
        pretrained_decoder: LatentDecoder,
        pretrained_embedding: nn.Embedding,
    ):
        super().__init__(config)
        self.config = config
        # replace encoder and decoder by LatentEncoder and LatentDecorer
        self.encoder = pretrained_encoder
        self.decoder = pretrained_decoder
        # link back to the shared embed_tokens coming from the pretrained encoder/decoder
        self.shared = pretrained_embedding
        self.z_dim = self.config.z_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> VAEOutput:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # skip encoder if text generation has already produced encoder outputs from context/query input
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,  # in BartModel encoder_hidden_states=encoder_outputs[0]
            latent_variable=encoder_outputs.latent_variable,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return VAEOutput(
            loss=encoder_outputs.loss,
            latent_variable=encoder_outputs.latent_variable,
            representation=encoder_outputs.representation,
            hidden_before_latent=encoder_outputs.hidden_before_latent,
            supp_data=encoder_outputs.supp_data,
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VAEForLM(BartForConditionalGeneration):

    def __init__(
        self,
        config: VAEConfigLM,
        pretrained: BartForConditionalGeneration,
        **kwargs
    ):
        super().__init__(config)
        self.gamma = self.config.gamma
        self.model: VAE = self._build_model(pretrained, config)
        self.lm_head = pretrained.lm_head

    @staticmethod
    def _build_model(pretrained, config):
        pretrained_encoder = pretrained.get_encoder()
        pretrained_decoder = pretrained.get_decoder()
        pretrained_embedding = pretrained.get_input_embeddings()
        return VAE(
            config,
            LatentEncoder(pretrained_encoder, config),
            LatentDecoder(pretrained_decoder, config),
            pretrained_embedding
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # trainable language model head
        logits = self.lm_head(outputs.last_hidden_state) #+ self.final_logits_bias
        supp_data = outputs.supp_data if outputs.supp_data is not None else {}

        # calculate composite loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_lm = self.gamma * loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            loss_z = outputs.loss  # loss on latent var
            loss = loss_lm + loss_z  # combine with language modelling loss
            supp_data['loss_lm'] = loss_lm  # keep track for plotting in TensorBoard
        else:
            loss = None
            supp_data['loss_lm'] = loss_lm = None
        return VAELMOutput(
            loss=loss,
            logits=logits,
            latent_variable=outputs.latent_variable,
            representation=outputs.representation,
            hidden_before_latent=outputs.hidden_before_latent,
            supp_data=supp_data,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Text generation will sample several examples.
        The inputs and encoder_outputs need therefore to be exanded in dim 0 to expand_size.
        The method from the inherited GenerationMixin parent class is overriden to also expand the latent variable!
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            # MODIFIED FROM generation_utils.GenerationMixin._expand_inputs_for_generation
            encoder_outputs["latent_variable"] = encoder_outputs.latent_variable.index_select(
                0, expanded_return_idx.to(encoder_outputs.latent_variable.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs


class MyPreTrainedModel(PreTrainedModel):

    """A bit of an unfortunate consequence of encoding twin examples as list
    instead of using an additional dimension of a tensor is that the PreTrainedModel
    class needs to be modified in the obscure method estimate_tokens() called by the 
    equally obscure floating_point_ops(). Using PreTrainedModel as base class is useful
    to be able to load model with from_pretrained().
    """

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name][0].numel() + input_dict[self.main_input_name][1].numel()
        else:
            logger.warn(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        # For models that inherit from [`PreTrainedModel`], uses that method to compute the number of
        #     floating point operations for every backward + forward pass. If using another model, either implement such a
        #     method in the model or subclass and override this method.

        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class Twin(MyPreTrainedModel):

    config_class = TwinConfig

    def __init__(
        self,
        config: TwinConfig,
        pretrained: BartModel
    ):
        super().__init__(config)
        pretrained_encoder = pretrained.get_encoder()
        # shared pretrained encoder
        self.encoders = nn.ModuleList([
            LatentEncoder(pretrained_encoder, config),
            LatentEncoder(pretrained_encoder, config),
        ])
        self.config = config
        self.mu = self.config.mu
        self.lambd = self.config.lambd

    def forward(
        self,
        input_ids: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ):
        outputs: List[LatentEncoderOutput] = [
            self.encoders[i](input_ids=input_ids[i], attention_mask=attention_mask[i], **kwargs)
            for i in range(len(input_ids))
        ]
        loss, loss_twin_z, loss_diag, loss_off_diag, cross_correl = self.all_losses(outputs)
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
            representations=[out.representation for out in outputs],
            hidden_before_latent=[out.hidden_before_latent for out in outputs],
            latent_variable=[out.latent_variable for out in outputs],
            supp_data=supp_data
        )

    def all_losses(self, outputs):
        loss_diag, loss_off_diag, cross_correl = compute_loss_on_twins([out.representation for out in outputs])
        losses = torch.stack([out.loss for out in outputs])
        losses = losses.sum()
        loss_twin_z = self.mu * (loss_diag + self.lambd * loss_off_diag)
        loss = losses + loss_twin_z
        return loss, loss_twin_z, loss_diag, loss_off_diag, cross_correl

    @staticmethod
    def update_supp_data(supp_data, outputs):
        for i, out in enumerate(outputs):
            for k, v in out.supp_data.items():
                supp_data[f"{k}_{i}"] = v
        return supp_data


class TwinSEQ2SEQ(Twin):

    def __init__(
        self,
        config: TwinLMConfig,
        pretrained: BartModel
    ):
        super().__init__(config, pretrained)
        pretrained_decoder = pretrained.get_decoder()
        # two LatentDecoders with the shared pretrained_decoder; only the decompression is independent
        self.decoders = nn.ModuleList([
            LatentDecoder(pretrained_decoder, config),
            LatentDecoder(pretrained_decoder, config),
        ])
        self.lm_head = pretrained.lm_head
        self.embedding = pretrained.get_input_embeddings()
        self.gamma = config.gamma

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = [
                    shift_tokens_right(
                        lbl, self.config.pad_token_id, self.config.decoder_start_token_id
                    )
                    for lbl in labels
                ]

        encoder_outputs: TwinOutput = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        decoder_outputs = [
           decoder(
                input_ids=decoder_input_ids[i],
                encoder_hidden_states=encoder_outputs.last_hidden_state[i],  # in BartModel encoder_hidden_states=encoder_outputs[0]
                latent_variable=encoder_outputs.latent_variable[i],
                # attention_mask=decoder_attention_mask[i],
                # encoder_attention_mask=attention_mask[i],
                # head_mask=decoder_head_mask[i],
                # cross_attn_head_mask=cross_attn_head_mask[i],
                # past_key_values=past_key_values[i],
                # inputs_embeds=decoder_inputs_embeds[i],
                # use_cache=use_cache,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                # return_dict=return_dict,
            ) for i, decoder in enumerate(self.decoders)
        ]

        # trainable language model head
        logits = [
            self.lm_head(out.last_hidden_state)
            for out in decoder_outputs
        ]
        supp_data = encoder_outputs.supp_data if encoder_outputs.supp_data is not None else {}

        # calculate composite loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_lm = self.gamma * sum([
                loss_fct(logits[i].view(-1, self.decoders[i].config.vocab_size), labels[i].view(-1))
                for i in range(len(input_ids))
            ])
            loss_z = encoder_outputs.loss  # loss on latent var
            loss = loss_lm + loss_z  # combine with language modelling loss
            supp_data['loss_lm'] = loss_lm  # keep track for plotting in TensorBoard
        else:
            loss = None
            supp_data['loss_lm'] = loss_lm = None

        return TwinLMOutput(
            loss=loss,
            logits=logits,
            representations=encoder_outputs.representations,
            supp_data=supp_data
        )


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


# https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch
# https://openreview.net/forum?id=Syx72jC9tm
# equi_2_to_1
class layer_2_to_1(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 5

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs):
        m = inputs.size(3)  # extract dimension

        ops_out = contractions_2_to_1(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)  # N x D x B x m

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m

        # bias
        output = output + self.bias

        return output


# ops_2_to_1
def contractions_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3)  # N x D x m

    sum_diag_part = torch.sum(diag_part, dim = 2).unsqueeze(dim = 2)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim = 3)  # N x D x m
    sum_of_cols = torch.sum(inputs, dim = 2)  # N x D x m
    sum_all = torch.sum(inputs, dim = (2, 3))  # N x D

    # op1 - (123) - extract diag
    op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    op2 = torch.cat([sum_diag_part for d in range(dim)], dim = 2)  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m

    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / (dim ** 2)

    return [op1, op2, op3, op4, op5]


class GraphEncoder(BartEncoder):

    def __init__(
        self,
        pretrained,
        config: GraphLatentConfig
    ):
        super().__init__(config)
        self.config = config
        self.num_nodes = self.config.num_nodes
        self.num_entity_features = self.config.num_entity_features
        self.flip_proba = config.flip_proba

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

        # Rosetta tensor as parameter
        rosetta = torch.empty(self.seq_length, self.d_encoder)
        nn.init.normal_(rosetta, std=0.02)
        # def NormalParameter(n_in, n_out, init_scale=1.0):
        #     """Parameter with random normal initialization"""
        #     w = torch.empty(n_in, n_out)
        #     nn.init.normal_(w, std=0.02 * init_scale)
        #     return nn.Parameter(w)
        self.rosetta = nn.Parameter(rosetta)
        self.attn = BartAttention(
            self.d_encoder,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )

        # adj matrix
        # latent vars
        self.hidden_features = self.config.hidden_features
        self.sampling_iterations = self.config.sampling_iterations
        self.latent_var_loss = self.config.latent_var_loss.split('-') if self.config.latent_var_loss is not None else None
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

    def forward(self, input_ids=None, **kwargs) -> LatentEncoderOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.model(input_ids=input_ids, **kwargs)
        x = encoder_outputs.last_hidden_state  # -> B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"observed seq length {length} mismatches with config.seq_length {self.seq_length} with input_ids.size()={input_ids.size()}"

        # remove sequence info through cross-attention
        # between a learn rosetta stone as magic query
        # encoder outputs whose attention values are combined in weighted sum
        # thus distroying sequence information but hopefully keeping
        # causal dependencies.
        # query: rosetta
        # key_state, value_states: encoder_hidden_states
        rosetta_with_batch_size = self.rosetta.data.repeat(batch_size, 1, 1)
        y, cross_attn_weights, cross_attn_present_key_value = self.attn(
            hidden_states=rosetta_with_batch_size,  # query
            key_value_states=x  # key and value
        )

        # compress
        y = self.vae_dropout(x)
        y = self.fc_compress(y)  # -> B x D x D (example: 32 example x 256 token x 256 hidden features)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        hidden_before_latent = y
        y = y.view(batch_size, (self.seq_length * self.hidden_features))

        # adj matrix
        adj = self.vae_dropout(y)
        adj = self.to_adj_matrix(adj)
        adj = self.norm_adj(adj)
        adj = self.act_fct(adj)
        adj = adj.view(-1, self.num_nodes, self.num_nodes)

        flipped = False
        if self.flip_proba > 0:
            p = random()
            if p <= self.flip_proba:
                adj = adj.transpose(-1, -2).contiguous()  # inverse 'causality', contiguous to allow view(-1, num_nodes ** 2)
                x = flip(x)  # -> B x L x H_enc
                encoder_outputs.hidden_states = flip(encoder_outputs.hidden_states)
                encoder_outputs.attentions = flip(encoder_outputs.attentions)
                flipped = True

        adj_matrix_representation = adj

        # entities
        entities = self.vae_dropout(y)
        entities = self.to_entity_embed(entities)
        entities = self.norm_entities(entities)
        entities = self.act_fct(entities)
        entities = entities.view(-1, self.num_entity_features, self.num_nodes)
        entities_representation = entities

        # permutation-independent sets
        # z_graph, z_entities = self.to_permutation_independent_set(adj, entities)
        z_graph, z_entities = adj, entities

        if self.latent_var_loss:
            loss, supp_data = self.compute_loss_on_latent_var(z_graph, z_entities, self.latent_var_loss)
        elif self.latent_var_loss is None:
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
            supp_data = {"loss_z": loss}
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        representation = [adj_matrix_representation, entities_representation]
        z_graph = adj.view(-1, self.num_nodes * self.num_nodes)
        z_entities = entities.view(-1, self.num_entity_features * self.num_nodes)
        z = torch.cat([z_graph, z_entities], -1)

        return LatentCGraphEncoderOutput(
            last_hidden_state=x,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            loss=loss,
            latent_variable=z,
            representation=representation,
            hidden_before_latent=hidden_before_latent,
            supp_data=supp_data,
            flipped=torch.tensor(flipped).cuda() if torch.cuda.is_available() else torch.tensor(flipped)
        )

    def to_permutation_independent_set(self, adj, entities, with_grad: bool = True):
        raise NotImplementedError

    def compute_loss_on_latent_var(self, z_graph, z_entities, include=['mmd', 'diag', 'sparse']):
        losses = {}
        if 'mmd' in include:
            with torch.no_grad():
                edge_sample, entity_sample = sample_graph(
                    self.num_nodes,
                    self.sample_num_entities,
                    self.sample_num_interactions,
                    self.num_entity_features,
                    self.sampling_iterations
                )
                if torch.cuda.is_available():
                    edge_sample = edge_sample.cuda()
                    entity_sample = entity_sample.cuda()
                z_graph_sample, z_entities_sample = self.to_permutation_independent_set(edge_sample.detach(), entity_sample.detach())

            losses['adj_matrix_distro_loss'] = self.alpha * mmd(
                z_graph_sample.view(self.sampling_iterations, self.num_nodes ** 2),
                z_graph.view(-1, self.num_nodes ** 2)
            )
            losses['entity_distro_loss'] = self.beta * mmd(
                z_entities_sample.view(self.sampling_iterations, self.num_nodes * self.num_entity_features),
                z_entities.view(-1,  self.num_nodes * self.num_entity_features)
            )

        if 'diag' in include:
            diag = z_graph.diagonal()
            loss_diag = diag ** 2
            losses['diag_loss'] = loss_diag.sum() / (self.num_nodes ** 2)  # num elements of diag scales as n

        if 'sparse' in include:
            losses['loss_adj_sparse'] = z_graph.abs().mean()
            losses['loss_node_sparse'] = z_entities.abs().mean()

        if 'DAG' in include:
            # # https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
            # # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
            # # naive (me...):
            d = self.num_nodes  # cosmetic
            W = z_graph  # cosmetic
            batch_size = W.size(0)
            I = torch.eye(d).unsqueeze(0).expand(batch_size, d, d)
            if torch.cuda.is_available():
                W = W.cuda()
                I = I.cuda()
            mat_power_d = torch.matrix_power(I + (W * W ) / d, d)  # based on below Yu et al
            trace = mat_power_d.diagonal(dim1=-1, dim2=-2).sum(-1)
            L_dag = trace - d
            losses['DAG'] = L_dag.mean()

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

        loss = sum(losses.values())
        supp_data = {
            "loss_z": loss,
            **losses,
        }
        return loss, supp_data


class GraphVAEForLM(VAEForLM):

    def __init__(
        self,
        config: GraphVAEConfigLM,
        pretrained: BartForConditionalGeneration,
        **kwargs
    ):
        super().__init__(config, pretrained,  **kwargs)

    @staticmethod
    def _build_model(pretrained, config):
        pretrained_encoder = pretrained.get_encoder()
        pretrained_decoder = pretrained.get_decoder()
        pretrained_embedding = pretrained.get_input_embeddings()
        return VAE(
            config,
            GraphEncoder(pretrained_encoder, config),
            LatentDecoder(pretrained_decoder, config),
            pretrained_embedding
        )


class CGraphVAEForLM(MyPreTrainedModel):

    def __init__(

        self,
        config: VAEConfigLM,
        pretrained: BartForConditionalGeneration,
        **kwargs
    ):
        super().__init__(config)
        self.gamma = self.config.gamma
        pretrained_encoder = pretrained.get_encoder()
        pretrained_decoder = pretrained.get_decoder()
        self.encoder = GraphEncoder(pretrained_encoder, config)
        self.decoder = LatentDecoder(pretrained_decoder, config)
        self.lm_head = pretrained.lm_head
        self.shared = pretrained.get_input_embeddings()
        self.z_dim = self.config.z_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = input_ids[0]  # unflipped input_ids
        attention_mask = attention_mask[0]  # collator prepares attention_mask for both unflipped and flipped input_ids
        # skip encoder if text generation has already produced encoder outputs from context/query input
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # important to do this AFTER the encoding to see if labels need to be flipped before shifting them right
        # to produce decoder_input_ids
        if labels is not None:
            unflipped_labels = labels[0]  # unflipped labels
            flipped_labels = labels[1]  # labels pre-flipped by loader
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                unflipped_decoder_input_ids = shift_tokens_right(
                    unflipped_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                flipped_decoder_input_ids = shift_tokens_right(
                    flipped_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # what/why does shift right insert decoder_start_token_id == eos_token_id???
        # https://github.com/huggingface/transformers/issues/20842
        # "bos_token_id": 0,
        # "eos_token_id": 2
        # "decoder_start_token_id": 2
        # "pad_token_id": 1
        # ^ = bos token
        # $ = eos token
        # § = decoder_start_token_id
        # + = padding token
        # unflipped => predict next token: causal language model
        # shift right labels for input to decoder
        # Right-shifted inputs: §^This is a cat.$+++++
        #                       ||||||||||||||||||||||
        # Original labels       ^This is a cat.$++++++

        # flipped => predict *previous* token: reverse causal language model
        # shift right flipped labels for input to decoder
        # Right-shifted flipped inputs: §$.tac a si sihT^+++++++
        #                               ||||||||||||||||||||||||
        # Flipped original labels:      $.tac a si sihT^++++++++

        if not encoder_outputs.flipped:
            decoder_outputs = self.decoder(
                input_ids=unflipped_decoder_input_ids,
                encoder_hidden_states=None,  # encoder_outputs.last_hidden_state,  # in BartModel encoder_hidden_states=encoder_outputs[0]
                latent_variable=encoder_outputs.latent_variable,
                attention_mask=decoder_attention_mask,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            adversarial_decoder_outputs = self.decoder(
                input_ids=flipped_decoder_input_ids,  # generated from flip(labels)!!  so labels are flipped!
                encoder_hidden_states=None,  # encoder_outputs.last_hidden_state,  # already flipped!
                latent_variable=encoder_outputs.latent_variable,
                attention_mask=flip(decoder_attention_mask),
                encoder_attention_mask=flip(attention_mask),
                head_mask=flip(decoder_head_mask),
                cross_attn_head_mask=flip(cross_attn_head_mask),
                past_key_values=flip(past_key_values),
                inputs_embeds=flip(decoder_inputs_embeds),
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # flipped causal model, encoder hidden states and labels are flipped
            # adj matrix is transposed

            decoder_outputs = self.decoder(
                input_ids=flipped_decoder_input_ids,  # generated from flip(labels)!!  so labels are flipped!
                encoder_hidden_states=None,  # encoder_outputs.last_hidden_state,  # already flipped!
                latent_variable=encoder_outputs.latent_variable,
                attention_mask=flip(decoder_attention_mask),
                encoder_attention_mask=flip(attention_mask),
                head_mask=flip(decoder_head_mask),
                cross_attn_head_mask=flip(cross_attn_head_mask),
                past_key_values=flip(past_key_values),
                inputs_embeds=flip(decoder_inputs_embeds),
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            adversarial_decoder_outputs = self.decoder(
                input_ids=unflipped_decoder_input_ids,
                encoder_hidden_states=None,  # encoder_outputs.last_hidden_state,  # in BartModel encoder_hidden_states=encoder_outputs[0]
                latent_variable=encoder_outputs.latent_variable,
                attention_mask=decoder_attention_mask,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # trainable language model head
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        adversarial_logits = self.lm_head(adversarial_decoder_outputs.last_hidden_state)
        supp_data = encoder_outputs.supp_data if encoder_outputs.supp_data is not None else {}

        # calculate composite loss

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if encoder_outputs.flipped:
                loss_lm = self.gamma * loss_fct(logits.view(-1, self.decoder.config.vocab_size), unflipped_labels.view(-1))
                loss_adv_lm = self.gamma * loss_fct(adversarial_logits.view(-1, self.decoder.config.vocab_size), flipped_labels.view(-1))
            else:
                loss_lm = self.gamma * loss_fct(logits.view(-1, self.decoder.config.vocab_size), flipped_labels.view(-1))
                loss_adv_lm = self.gamma * loss_fct(adversarial_logits.view(-1, self.decoder.config.vocab_size), unflipped_labels.view(-1))

            loss_lm = loss_lm / loss_adv_lm
            loss_z = encoder_outputs.loss  # loss on latent var
            loss = loss_lm + loss_z  # combine with language modelling loss
            supp_data['loss_lm'] = loss_lm  # keep track for plotting in TensorBoard
        else:
            loss = None
            supp_data['loss_lm'] = loss_lm = None

        return CGVAELMOutput(
            loss=loss,
            logits=logits,
            latent_variable=encoder_outputs.latent_variable,
            representation=encoder_outputs.representation,
            hidden_before_latent=encoder_outputs.hidden_before_latent,
            supp_data=supp_data,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            flipped=encoder_outputs.flipped,
        )


class BartFlip(MyPreTrainedModel):

    def __init__(
        self,
        config: FlipBartConfig,
        pretrained: BartForConditionalGeneration,
        **kwargs
    ):
        super().__init__(config)
        # self.encoder = pretrained.get_encoder()
        self.encoder = FlippableBartEncoder(config)
        self.decoder = pretrained.get_decoder()
        self.freeze_pretrained = self.config.freeze_pretrained
        # freeze the pretrained model
        if self.freeze_pretrained in ['both', 'encoder']:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        # if self.freeze_pretrained in ['both', 'decoder']:
        #     for param in self.decoder.parameters():
        #         param.requires_grad_(False)
        if self.freeze_pretrained is not None and self.freeze_pretrained not in ['both', 'encoder', 'decoder']:
            raise ValueError(f"not sure what to freeze or not with freeze_pretrained={self.freeze_pretrained}")

        # self.middle_flip_layers = nn.ModuleList([FlippableBartEncoderLayer(config) for _ in range(config.num_flip_layers)])
        self.lm_head = pretrained.lm_head
        self.shared = pretrained.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        unflipped_input_ids = input_ids[0]  # unflipped input_ids
        flipped_input_ids = input_ids[1]  # flipped input_ids
        unflipped_attention_mask = attention_mask[0]  # collator prepares attention_mask for both unflipped and flipped input_ids
        flipped_attention_mask = attention_mask[1]  # collator prepares attention_mask for both unflipped and flipped input_ids
        # skip encoder if text generation has already produced encoder outputs from context/query input
                # to produce decoder_input_ids
        if labels is not None:
            unflipped_labels = labels[0]  # unflipped labels
            flipped_labels = labels[1]  # flipped labels

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            unflipped_decoder_input_ids = shift_tokens_right(
                unflipped_labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            flipped_decoder_input_ids = shift_tokens_right(
                flipped_labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        flip = random() < 0.5
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=unflipped_input_ids,
                attention_mask=unflipped_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                flip=flip
            )

        # what/why does shift right insert decoder_start_token_id == eos_token_id???
        # https://github.com/huggingface/transformers/issues/20842
        # "bos_token_id": 0,
        # "eos_token_id": 2
        # "decoder_start_token_id": 2
        # "pad_token_id": 1
        # ^ = bos token
        # $ = eos token
        # § = decoder_start_token_id
        # + = padding token
        # unflipped => predict next token: causal language model
        # shift right labels for input to decoder
        # Right-shifted inputs: §^This is a cat.$+++++
        #                       ||||||||||||||||||||||
        # Original labels       ^This is a cat.$++++++

        # flipped => predict *previous* token: reverse causal language model
        # shift right flipped labels for input to decoder
        # Right-shifted flipped inputs: §$.tac a si sihT^+++++++
        #                               ||||||||||||||||||||||||
        # Flipped original labels:      $.tac a si sihT^++++++++

        hidden_states = encoder_outputs[0]

        if self.freeze_pretrained in ['both', 'encoder']:
            hidden_states.requires_grad_(True)

        # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     # dtype = (self.encoder.embed_tokens(unflipped_input_ids) * self.encoder.embed_scale).dtype
        #     dtype = torch.float32
        #     expanded_attention_mask = _expand_mask(unflipped_attention_mask, dtype)

        # for idx, encoder_flip_layer in enumerate(self.middle_flip_layers):
        #     layer_outputs = encoder_flip_layer(
        #         hidden_states,
        #         expanded_attention_mask,
        #         output_attentions=output_attentions,
        #         # layer_head_mask=head_mask[idx] if head_mask is not None else None,
        #         flip=flip
        #     )
        #     hidden_states = layer_outputs[0]

        if flip:
            labels = flipped_labels
            decoder_input_ids = flipped_decoder_input_ids
            attention_mask = flipped_attention_mask
            # hidden_states = hidden_states.flip(1)  # not sure about that...
        else:
            labels = unflipped_labels
            decoder_input_ids = unflipped_decoder_input_ids
            attention_mask = unflipped_attention_mask

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,  # encoder_outputs.last_hidden_state,  # in BartModel encoder_hidden_states=encoder_outputs[0]
            encoder_attention_mask=attention_mask,
            return_dict=return_dict
        )

        # trainable language model head
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        # supp_data = encoder_outputs.supp_data if encoder_outputs.supp_data is not None else {}
        supp_data = {}

        # calculate losss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Could add losses on the attention weights (sparsity, diganoal off diagnoal, DAG structure)
            loss = loss_fct(logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            # supp_data['loss_lm'] = loss_lm  # keep track for plotting in TensorBoard
        else:
            loss = None
            # supp_data['loss_lm'] = loss_lm = None

        return CGVAELMOutput(
            loss=loss,
            logits=logits,
            supp_data=supp_data,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            flipped=torch.tensor(flip).cuda() if torch.cuda.is_available() else torch.tensor(flip)
        )
