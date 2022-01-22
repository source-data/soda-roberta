from itertools import product
from random import sample, gauss
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
from transformers import (
    BartConfig, BartTokenizerFast, BartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, BaseModelOutputWithPastAndCrossAttentions
# from transformers.utils import logging
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def compute_kernel(x, y):
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


def mmd(x, y):
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def sample_entity_subset(N_max, mu=3, sigma=0):
    N = min(N_max, round(gauss(mu, sigma)))
    entities = sample(list(range(N_max)), k=N)
    return entities


def sample_interactions(N_max, N_entities, N_interactions):
    entity_subset = sample_entity_subset(N_max, mu=N_entities)
    comb = list(product(entity_subset, entity_subset))
    z = torch.zeros(N_max, N_max)
    num_interactions = min(round(gauss(mu=N_interactions, sigma=0)), len(entity_subset)**2)
    try:
        coord = sample(comb, k=num_interactions)
    except ValueError as e:
        print(e)
        print(f"k={num_interactions}, entity_subset={entity_subset}")
        return
    z[list(zip(*coord))] = 1.0
    return z, entity_subset


def sample_node_labels(N_entities, entity_subset, M_features=10):
    p = torch.full([N_entities, M_features], 0.5)
    u = torch.bernoulli(p)
    v = torch.zeros(N_entities, M_features)
    v[entity_subset] = u[entity_subset]
    return v


def interaction_samples(N_max, N_entities, N_interactions, M_features, iterations=10):
    # TODO: plot distribution of mmd distance bewteen random pairs
    interactions = []
    node_labels = []
    for i in range(iterations):
        z1, entity_subset = sample_interactions(N_max, N_entities, N_interactions)
        interactions.append(z1.unsqueeze(0))
        z2 = sample_node_labels(N_max, entity_subset, M_features)
        node_labels.append(z2.unsqueeze(0))
    interactions = torch.cat(interactions, 0)
    node_labels = torch.cat(node_labels, 0)
    return interactions, node_labels


class BecauseConfig(BartConfig):
    def __init__(self, max_nodes=None, num_node_features=None, *args, **kwargs):
        super(BecauseConfig).__init__(*args, **kwargs)
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features


@dataclass
class BecauseOutput(MaskedLMOutput):
    z_1: torch.Tensor = None
    z_2: torch.Tensor = None
    supp_data: torch.Tensor = None  # custom data for logging


class Because(nn.Module):

    def __init__(
        self,
        pretrained: BartForConditionalGeneration,
        freeze_pretrained=True,
        max_nodes=None,
        num_node_features=None,
        num_entities=3,
        num_interactions=3,
        sampling_iterations=10,
        seq_length=1024
    ):
        super().__init__()
        # from the pretrained model
        self.pretrained = pretrained
        self.freeze_pretrained = freeze_pretrained
        self.encoder = self.pretrained.get_encoder()
        self.decoder = self.pretrained.get_decoder()
        # freeze the pretrained encoder and decoder
        if self.freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            for param in self.decoder.parameters():
                param.requires_grad_(False)
        self.lm_head = self.pretrained.lm_head
        self.d_encoder = self.encoder.config.d_model
        self.d_decoder = self.decoder.config.d_model
        self.seq_length = seq_length
        self.pad_token_id = self.decoder.config.pad_token_id
        self.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        # latent vars
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features
        self.num_entities = num_entities
        self.num_interactions = num_interactions
        self.hidden_features = 100
        self.sampling_iterations = sampling_iterations
        self.z1_dim = self.max_nodes ** 2
        self.z2_dim = self.max_nodes * self.num_node_features
        # own layers
        self.act_fct = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.fc_compress_1 = nn.Linear(self.d_encoder, 3 * self.hidden_features)
        self.fc_compress_2 = nn.Linear(3 * self.hidden_features, self.hidden_features)
        self.trfm_1 = nn.TransformerEncoderLayer(d_model=self.d_encoder, nhead=8)
        self.fc_z1 = nn.Linear(self.seq_length * self.hidden_features, self.z1_dim)
        self.fc_z2 = nn.Linear(self.seq_length * self.hidden_features, self.z2_dim)
        self.fc_dec_1 = nn.Linear(self.z1_dim, self.seq_length * self.hidden_features)
        self.fc_dec_2 = nn.Linear(self.z2_dim, self.seq_length * self.hidden_features)
        self.fc_dec_3 = nn.Linear(2, 1)
        self.fc_decompress_1 = nn.Linear(self.hidden_features, 3 * self.hidden_features)
        self.fc_decompress_2 = nn.Linear(3 * self.hidden_features, self.d_decoder)
        self.trfm_2 = nn.TransformerEncoderLayer(d_model=self.hidden_features, nhead=10)

    def forward(self, input_ids=None, labels=None, **kwargs) -> BecauseOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.encoder(input_ids=input_ids, **kwargs)
        y = encoder_outputs[0]
        y.requires_grad_(True)
        batch_size, length, hidden_size = y.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"{length} <> {self.seq_length}"
        assert hidden_size == self.d_encoder, f"{hidden_size} <> {self.d_encoder}"

        # transformer and compression
        y = self.trfm_1(y)  # -> B x L x H_enc (example: 32 x 512 x 768)
        y = self.fc_compress_1(y)  # -> B x L x 3*H (example: 32 x 512 x 300)
        y = self.act_fct(y)
        y = self.fc_compress_2(y)  # -> B x L x H (example: 32 x 512 x 100)
        y = self.act_fct(y)

        # # to adj matrix latent variable
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 51_200)
        z_1 = self.fc_z1(y)  # -> B x Z_1  (example: 32 x 10**2)
        z_2 = self.fc_z2(y)  # -> B x Z_2  (example: 32 x 10*100)
        z_1 = self.act_fct(z_1)
        z_2 = self.act_fct(z_2)
        # adj matrix and entity embeddings to decoder input
        z_1 = self.dropout(z_1)
        y_1 = self.fc_dec_1(z_1)  # -> B x (L * H)
        y_1 = self.dropout(y_1)

        z_2 = self.dropout(z_2)
        y_2 = self.fc_dec_2(z_2)  # -> B x (L * H)
        y_2 = self.dropout(y_2)

        # introduce third dimension to concatenate both latent var
        y_1 = y_1.unsqueeze(-1)  # -> B x (L * H) x 1
        y_2 = y_2.unsqueeze(-1)  # -> B x (L * H) x 1
        y = torch.cat([y_1, y_2], -1)  # -> B x (L * H) x 2

        # restore dimensions
        y = self.fc_dec_3(y)  # -> B x (L * H) x 1
        y = y.squeeze(-1)  # -> B x (L * H)
        y = y.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H

        # transformer layer and decompression
        y = self.trfm_2(y)
        y = self.fc_decompress_1(y)  # -> B x L x 3*H
        y = self.act_fct(y)
        y = self.fc_decompress_2(y)  # -> B x L x H_dec
        y = self.act_fct(y)

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

        # trainable language model head
        lm_logits = self.lm_head(decoder_outputs[0])  # maybe not necessary

        # calculate composite loss
        if labels is not None:
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            with torch.no_grad():
                adj_sampling, node_label_sampling = interaction_samples(self.max_nodes, self.num_entities, self.num_interactions, self.num_node_features, self.sampling_iterations)
            adj_matrix_distro_loss = mmd(adj_sampling.view(self.sampling_iterations, self.max_nodes ** 2), z_1)
            node_label_distro_lost = mmd(node_label_sampling.view(self.sampling_iterations, self.max_nodes * self.num_node_features), z_2)
            adj_matrix_distro_loss = 50 * adj_matrix_distro_loss
            node_label_distro_lost = 50 * node_label_distro_lost
            loss = masked_lm_loss + adj_matrix_distro_loss + node_label_distro_lost
            supp_data = torch.cat([
                masked_lm_loss.unsqueeze_(0),
                adj_matrix_distro_loss.unsqueeze_(0),
                node_label_distro_lost.unsqueeze_(0)
            ], 0).unsqueeze(0)
        else:
            loss = None
            supp_data = None

        return BecauseOutput(
            loss=loss,
            logits=lm_logits,
            # supplementarty data for logging only
            supp_data=supp_data,
            z_1=z_1,
            z_2=z_2,
            # past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,  'BaseModelOutput' object has no attribute 'last_hidden_states'
            # encoder_hidden_states=encoder_outputs.hidden_states,
        )


def self_test():
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    test_string = "Huggingface office is based in "
    inputs = tokenizer([test_string])  # , return_tensors="pt")
    inputs = tokenizer.pad(inputs, return_tensors="pt", pad_to_multiple_of=512)
    input_ids = inputs["input_ids"]
    seq2seq = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    because = Because(
        pretrained=seq2seq,
        max_nodes=4,
        num_node_features=10,
        sampling_iterations=10,
        seq_length=input_ids.size(-1)
    )
    # input_ids[input_ids == -100] = seq2seq.config.pad_token_id
    # import pdb; pdb.set_trace()
    # labels = input_ids.masked_fill_(input_ids == seq2seq.config.pad_token_id, -100)
    labels = input_ids.clone()
    labels[labels == seq2seq.config.pad_token_id] = -100
    outputs = because(input_ids=input_ids, labels=labels)
    output_ids = torch.argmax(torch.softmax(outputs.logits, -1), -1)
    result_string = tokenizer.decode(output_ids[0].tolist())
    print(result_string)


def main():
    self_test()


if __name__ == "__main__":
    self_test()
