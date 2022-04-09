from itertools import product
import pdb
from random import sample, gauss
from dataclasses import dataclass
from typing import List, Dict
import torch
from torch import nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
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
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    # x = x.cpu()
    # y = y.cpu()
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    # if torch.cuda.is_available():
    #     mmd = mmd.cuda()
    return mmd


def sample_z(z_dim: int, iterations: int = 100) -> torch.Tensor:
    return torch.randn(200, z_dim)


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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_pretrained = freeze_pretrained
        self.hidden_features = hidden_features
        self.z_dim = z_dim
        self.sampling_iterations = sampling_iterations
        self.seq_length = seq_length
        self.residuals = residuals


class VAEConfigForTokenClassification(VAEConfig):

    def __init__(self, classifier_dropout: float = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_dropout = classifier_dropout


@dataclass
class TwinVAEConfig(VAEConfigForTokenClassification):

    def __init__(self, lambd_a: float = None, **kwargs):
        super().__init__(**kwargs)
        self.lambd_a = lambd_a  # not a typo; weight on off diagnonal temrs of twin loss


@dataclass
class VAEOutput(MaskedLMOutput):
    supp_data: Dict[str, torch.Tensor] = None
    z: torch.Tensor = None


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
        # freeze the pretrained encoder and decoder
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
        # own layers
        self.act_fct = nn.GELU()
        self.vae_dropout = nn.Dropout(p=config.dropout)
        self.fc_compress = nn.Linear(self.d_encoder, self.hidden_features)
        self.norm_compress = nn.LayerNorm(self.hidden_features, elementwise_affine=False)
        self.fc_z_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_dim)
        self.norm_z = nn.LayerNorm(self.z_dim, elementwise_affine=False)
        self.fc_z_2 = nn.Linear(self.z_dim, self.seq_length * self.hidden_features)
        self.norm_decompress = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)
        self.fc_decompress = nn.Linear(self.hidden_features, self.d_decoder)

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.encoder(input_ids=input_ids, **kwargs)
        x = encoder_outputs[0]  # B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        # compress
        y = x  # keep x as residual
        y = self.vae_dropout(y)
        y = self.fc_compress(y)  # -> B x L x H (example: 32 x 512 x 100)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 51_200)
        # first latent var
        y = self.vae_dropout(y)
        z = self.fc_z_1(y)  # -> B x Z_1
        z = self.norm_z(z)
        z = self.act_fct(z)
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
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=y,
            **kwargs
        )

        logits = decoder_outputs[0]

        loss = self.compute_loss_on_latent_var(z)

        return VAEOutput(
            loss=loss,
            logits=logits,
            z=z,
            supp_data={}
        )

    def compute_loss_on_latent_var(self, z) -> torch.Tensor:
        with torch.no_grad():
            z_samples = sample_z(self.z_dim, self.sampling_iterations)
            z_loss = mmd(z_samples, z)
        return z_loss


class VAEForMaskedLM(VAE):

    def __init__(self, pretrained, config: VAEConfig, **kwargs):
        super().__init__(pretrained, config, **kwargs)
        self.model = VAE(pretrained, config)
        self.lm_head = nn.Linear(self.pretrained.config.d_model, self.pretrained.model.shared.num_embeddings, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs) -> VAEOutput:
        outputs = self.model(input_ids, **kwargs)
        # outputs = super().forward(input_ids, labels, **kwargs)

        # trainable language model head
        logits = self.lm_head(outputs.logits)
        supp_data = outputs.supp_data

        # calculate composite loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_lm = loss_fct(logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            loss = outputs.loss + loss_lm
            supp_data['loss_lm'] = loss_lm
        else:
            loss = None
            supp_data['loss_lm'] = loss_lm = None
        return VAEOutput(
            loss=loss,
            logits=logits,
            z=outputs.z,
            supp_data=supp_data
        )


class TwinVAEForMaskedLM(nn.Module):

    def __init__(
        self,
        models: List[VAEForMaskedLM],
        config: TwinVAEConfig,
        **kwargs
    ):
        super().__init__()
        # TODO: check PyTorch mechanics how to register properly arbitrary list of models
        self.model_1 = models[0]
        self.model_2 = models[1]
        self.config = config

    def forward(
        self,
        input_ids: List[torch.Tensor] = None,
        labels: List[torch.Tensor] = None,
        attention_mask: List[torch.Tensor] = None,
        **kwargs
    ):
        # outputs = [
        #     model(input_ids=input_ids[i], labels=labels[i], attention_mask=attention_mask[i], **kwargs)
        #     for i, model in enumerate([self.model_1, self.model_2])
        # ]
        outputs = [
            self.model_1(input_ids=input_ids[:, :, 0], labels=labels[:, :, 0], attention_mask=attention_mask[:, :, 0], **kwargs),
            self.model_2(input_ids=input_ids[:, :, 1], labels=labels[:, :, 1], attention_mask=attention_mask[:, :, 1], **kwargs)
        ]

        loss_twin_z = self.compute_loss_on_twin_latent_vars([out.z for out in outputs])
        losses = torch.stack([out.loss for out in outputs])
        loss = losses.sum() + loss_twin_z

        return TwinOutput(
            logits=[out.logits for out in outputs],
            loss=loss,
            supp_data={
                "loss_twin_z": loss_twin_z
            }
        )

    def compute_loss_on_twin_latent_vars(self, z: List[torch.Tensor]) -> torch.Tensor:
        assert len(z) == 2  # for the moment, this works only on twin pairs, not for higher order
        assert len(z[0]) == len(z[1])  # square
        batch_size = len(z[0])
        c = (z[0].T @ z[1]) / batch_size
        diag = c.diagonal()
        off_diag = c - torch.diag_embed(diag)
        loss_diag = (diag - 1) ** 2
        loss_off_diag = off_diag ** 2
        loss = loss_diag.sum() + self.config.lambd_a * loss_off_diag.sum()
        return loss
