import pdb
from dataclasses import dataclass
from typing import List, Dict, Union
from sklearn.multiclass import OutputCodeClassifier
import torch
from torch import nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartModel,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import (
    BaseModelOutput, MaskedLMOutput,
    BaseModelOutputWithPastAndCrossAttentions
)
# from transformers.utils import logging
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
    except RuntimeError:
        print(f"tiled_x.device={tiled_x.device}")
        print(f"tiled_y.device={tiled_y.device}")
        raise Exception()
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


class VAEConfig(BartConfig):

    # inherited from BartConfig
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
        residuals: bool = True,
        latent_var_loss: str = 'mmd',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_pretrained = freeze_pretrained
        self.hidden_features = hidden_features
        self.z_dim = z_dim
        self.sampling_iterations = sampling_iterations
        self.seq_length = seq_length
        self.residuals = residuals
        self.latent_var_loss = latent_var_loss


class VAEConfigLM(VAEConfig):

    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # weights of lm loss when composed with loss on latent var z


class VAEConfigForTokenClassification(VAEConfig):

    def __init__(self, classifier_dropout: float = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_dropout = classifier_dropout


@dataclass
class TwinVAEConfig(VAEConfigLM):

    def __init__(self, lambd_a: float = None, mu: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd_a = lambd_a  # not a typo; weight on off diagonal terms of twin loss
        self.mu = mu  # weight twin z loss vs the other losses


@dataclass
class VAEOutput(MaskedLMOutput):
    supp_data: Dict[str, torch.Tensor] = None
    z: torch.Tensor = None
    representation: torch.Tensor = None


@dataclass
class TwinOutput(MaskedLMOutput):
    supp_data: Dict[str, torch.Tensor] = None
    # outputs: List[VAEOutput] = None


class VAE(nn.Module):

    def __init__(
        self,
        pretrained: BartForConditionalGeneration,
        config: VAEConfig
    ):
        super().__init__()
        self.config = config
        # from the pretrained model
        self.pretrained = pretrained
        self.freeze_pretrained = self.config.freeze_pretrained
        self.encoder = self.pretrained.get_encoder()
        self.decoder = self.pretrained.get_decoder()
        # freeze the pretrained encoder and/or decoder
        if self.freeze_pretrained == 'both':
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            for param in self.decoder.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained == 'encoder':
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained == 'decoder':
            for param in self.decoder.parameters():
                param.requires_grad_(False)
        elif self.freeze_pretrained is None or self.freeze_pretrained == '':
            pass
        else:
            raise ValueError(f"not sure what to freeze or not with freeze_pretrained={self.freeze_pretrained}")

        self.d_encoder = self.encoder.config.d_model
        self.d_decoder = self.decoder.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.decoder.config.pad_token_id
        self.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        self.residuals = self.config.residuals
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
        self.fc_z_2 = nn.Linear(self.z_dim, self.seq_length * self.hidden_features)
        self.norm_decompress = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)
        self.fc_decompress = nn.Linear(self.hidden_features, self.d_decoder)

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.encoder(input_ids=input_ids, **kwargs)
        x = encoder_outputs[0]  # -> B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"observed seq length {length} mismatches with config.seq_length {self.seq_length} with input_ids.size()={input_ids.size()}"
        # compress
        y = x  # keep x for later as residual
        y = self.vae_dropout(y)
        y = self.fc_compress(y)  # -> B x L x H (example: 32 x 512 x 100)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 51_200)
        # latent var
        y = self.vae_dropout(y)
        if self.latent_var_loss == "mmd" or self.latent_var_loss is None:
            z = self.fc_z_1(y)  # -> B x Z
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
        # decompress
        # y = self.act_fct(z)  # beneficial?
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
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=y,
            **kwargs
        )

        logits = decoder_outputs[0]

        if self.latent_var_loss == "mmd": 
            loss = self.compute_mmd_loss(z)
        elif self.latent_var_loss == "kl":
            loss = self.compute_kl_loss(z_mean, z_logvar)
            # loss = float(1/(1 + exp(-k * (step - x0))))  #  would need access to training_step, modify Trainer class
        elif self.latent_var_loss is None:
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
        else:
            raise ValueError(f"unknown loss type on latent variable {self.latent_var_loss}")

        return VAEOutput(
            loss=loss,
            logits=logits,
            z=z,
            representation=representation,
            supp_data={"loss_z": loss}
        )

    def compute_mmd_loss(self, z) -> torch.Tensor:
        # https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        z_samples = sample_z(self.z_dim, self.sampling_iterations)
        z_loss = mmd(z_samples, z)
        return z_loss

    def compute_kl_loss(self, mean, logvar) -> torch.Tensor:
        # https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        return kl.sum()


class VAEForLM(VAE):

    def __init__(self, pretrained: BartModel, config: VAEConfigLM, **kwargs):
        super().__init__(pretrained, config, **kwargs)
        self.gamma = config.gamma
        self.model = VAE(pretrained, config)
        self.lm_head = nn.Linear(self.pretrained.config.d_model, self.pretrained.shared.num_embeddings, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        outputs = self.model(input_ids, **kwargs)
        # outputs = super().forward(input_ids, labels, **kwargs)

        # trainable language model head
        logits = self.lm_head(outputs.logits)
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
        return VAEOutput(
            loss=loss,
            logits=logits,
            z=outputs.z,
            representation=outputs.representation,
            supp_data=supp_data
        )


class TwinVAEForLM(nn.Module):

    def __init__(
        self,
        models: Union[VAEForLM, List[VAEForLM]],
        config: TwinVAEConfig
    ):
        super().__init__()
        self.models = nn.ModuleList(models) if isinstance(models, list) else nn.ModuleList([models])
        self.config = config
        self.z_dims = [m.z_dim for m in self.models]
        self.lambd_a = self.config.lambd_a
        self.mu = self.config.mu
        # from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # does not seem to help much when tested on trivial same-same control dataset
        # projectors = []
        # for i in range(len(self.models)):
        #     projectors.append(nn.Sequential(
        #         nn.Linear(self.z_dims[i], 2 * self.z_dims[i], bias=False),
        #         nn.BatchNorm1d(2 * self.z_dims[i], affine=False),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(2 * self.z_dims[i], self.z_dims[i], bias=False),
        #         nn.BatchNorm1d(self.z_dims[i], affine=False))
        #     )
        # self.projectors = nn.ModuleList(projectors)

    def forward(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ):
        if len(self.models) == 1:  # single model for all twin examples
            outputs = [
                self.models[0](input_ids=input_ids[i], labels=labels[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
            # z = [self.projectors[0](out.z) for i, out in enumerate(outputs)]
        else:  # one model trained for each type fo twin example
            outputs = [
                self.models[i](input_ids=input_ids[i], labels=labels[i], attention_mask=attention_mask[i], **kwargs)
                for i in range(len(input_ids))
            ]
            # z = [self.projectors[i](out.z) for i, out in enumerate(outputs)]
        loss_diag, loss_off_diag = self.compute_loss_on_twins([out.representation for out in outputs])
        losses = torch.stack([out.loss for out in outputs])
        loss_twin_z = self.mu * (loss_diag + self.lambd_a * loss_off_diag)
        loss = losses.sum() + loss_twin_z
        return TwinOutput(
            loss=loss,
            logits=[out.logits for out in outputs],
            supp_data={
                "loss_diag": loss_diag,
                "loss_off_diag": loss_off_diag,
                "loss_twin_z": loss_twin_z,
                "loss_z_1": outputs[0].supp_data["loss_z"],
                "loss_z_2": outputs[1].supp_data["loss_z"],
                "loss_lm_1": outputs[0].supp_data["loss_lm"],
                "loss_lm_2": outputs[1].supp_data["loss_lm"],
            }
        )

    def compute_loss_on_twins(self, z: List[torch.Tensor]) -> torch.Tensor:
        assert len(z) == 2, "for the moment, this works only on twin pairs, not for higher order"
        assert z[0].size() == z[1].size(), "z dims have to be equal for square cross correl matrix"
        z = [t.cpu() for t in z]
        batch_size, z_dim = z[0].size()
        c = (z[0].T @ z[1]) / batch_size
        diag = c.diagonal()
        off_diag = c - torch.diag_embed(diag)
        # geeky way for off_diag https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # re-order matrix with 1-element longer rows such that first column is the diag
        # off_diag = c.flatten()[:-1].view(z_dim - 1, z_dim + 1)[:, 1:].flatten()
        loss_diag = (diag - 1) ** 2
        loss_off_diag = off_diag ** 2
        loss_diag = loss_diag.sum() / z_dim
        loss_off_diag = loss_off_diag.sum() / (z_dim ** 2)
        if torch.cuda.is_available():
            loss_diag = loss_diag.cuda()
            loss_off_diag = loss_off_diag.cuda()
        return loss_diag, loss_off_diag
