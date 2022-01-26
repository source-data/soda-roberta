from itertools import product
from random import sample, gauss
from dataclasses import dataclass
from typing import Tuple, List
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


def sample_node_subset(max_num_nodes: int, avg_num_nodes: float = 3.0, sigma: float = 0) -> List[int]:
    num_nodes = min(max_num_nodes, round(gauss(avg_num_nodes, sigma)))
    node_subset = sample(list(range(max_num_nodes)), k=num_nodes)
    return node_subset


def sample_edges(max_num_nodes, node_subset, avg_num_interactions, sigma: float = 0):
    all_pairwise_interactions = list(product(node_subset, node_subset))
    adj_matrix = torch.zeros(max_num_nodes, max_num_nodes)
    max_num_interactions = len(node_subset) ** 2
    num_interactions = min(round(gauss(mu=avg_num_interactions, sigma=sigma)), max_num_interactions)
    try:
        pairwise_interactions = sample(all_pairwise_interactions, k=num_interactions)
    except ValueError as e:
        print(e)
        print(f"k={num_interactions}, entity_subset={node_subset}")
        return
    adj_matrix[list(zip(*pairwise_interactions))] = 1.0
    return adj_matrix


def sample_node_labels(max_num_nodes, node_subset, node_features=10):
    p = torch.full([max_num_nodes, node_features], 0.5)
    u = torch.bernoulli(p)
    v = torch.zeros(max_num_nodes, node_features)
    v[node_subset] = u[node_subset]
    return v


def sample_graph(max_num_nodes, num_entities, num_interactions, num_node_features, iterations=10):
    # TODO: plot distribution of mmd distance bewteen random pairs
    edges = []
    node_embeddings = []
    for i in range(iterations):
        node_subset = sample_node_subset(max_num_nodes, avg_num_nodes=num_entities)
        adj_matrix = sample_edges(max_num_nodes, node_subset, num_interactions)
        edges.append(adj_matrix.unsqueeze(0))
        labeled_nodes = sample_node_labels(max_num_nodes, node_subset, num_node_features)
        node_embeddings.append(labeled_nodes.unsqueeze(0))
    edges = torch.cat(edges, 0)
    node_embeddings = torch.cat(node_embeddings, 0)
    return edges, node_embeddings


class BecauseConfig(BartConfig):
    def __init__(
        self,
        freeze_pretrained: str = 'both',
        hidden_features: int = 100,
        num_nodes: int = 10,
        num_node_features: int = 10,
        num_edge_features:int = 10,
        sample_num_entities: int = 10,
        sample_num_interactions: int = 10,
        sample_num_interaction_types = 3,
        sampling_iterations: int = 100,
        seq_length: int = 512,
        alpha: float = 1E05,
        beta: float = 1E05,
        *args, **kwargs
    ):
        super(BecauseConfig).__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.freeze_pretrained = freeze_pretrained
        self.hidden_features = hidden_features
        self.sample_num_entities = sample_num_entities
        self.sample_num_interactions = sample_num_interactions
        self.sample_num_interaction_types = sample_num_interaction_types
        self.sampling_iterations = num_node_features
        self.seq_length = seq_length
        self.alpha = alpha
        self.beta = beta


@dataclass
class BecauseOutput(MaskedLMOutput):
    z: torch.Tensor = None
    supp_data: torch.Tensor = None  # custom data for logging


class Because(nn.Module):

    def __init__(self, pretrained: BartForConditionalGeneration, config: BecauseConfig):
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
        elif self.freeze_pretrained is None:
            pass
        else:
            raise ValueError(f"not sure what to freeze or not with freeze_pretrained={self.freeze_pretrained}")
        self.lm_head = self.pretrained.lm_head
        self.d_encoder = self.encoder.config.d_model
        self.d_decoder = self.decoder.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.decoder.config.pad_token_id
        self.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        # latent vars
        self.num_nodes = self.config.num_nodes
        self.num_node_features = self.config.num_node_features
        self.num_edge_features = self.config.num_edge_features
        self.sample_num_entities = self.config.sample_num_entities
        self.sample_num_interactions = self.config.sample_num_interactions
        self.sample_num_interaction_types = self.config.sample_num_interaction_types
        self.hidden_features = self.config.hidden_features
        self.sampling_iterations = self.config.sampling_iterations
        self.z_1_dim = (self.num_nodes ** 2)  # * self.num_edge_features
        self.z_2_dim = self.num_nodes * self.num_node_features
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        # own layers
        self.act_fct = nn.GELU()
        self.loss_fct = nn.CrossEntropyLoss()

        self.fc_compress = nn.Linear(self.d_encoder, self.hidden_features)
        self.norm_compress = nn.LayerNorm(self.hidden_features, elementwise_affine=False)

        self.fc_z_1_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_1_dim)
        self.norm_z_1 = nn.LayerNorm(self.z_1_dim, elementwise_affine=False)
        self.fc_z_1_2 = nn.Linear(self.z_1_dim, self.seq_length * self.hidden_features)
        self.norm_decompress_1 = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)

        self.fc_z_2_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_2_dim)
        self.norm_z_2 = nn.LayerNorm(self.z_2_dim, elementwise_affine=False)
        self.fc_z_2_2 = nn.Linear(self.z_2_dim, self.seq_length * self.hidden_features)
        self.norm_decompress_2 = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)

        self.fc_decompress = nn.Linear(2 * self.hidden_features, self.d_decoder)

    def forward(self, input_ids=None, labels=None, **kwargs) -> BecauseOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.encoder(input_ids=input_ids, **kwargs)
        x = encoder_outputs[0]  # B x L x H_enc
        if self.freeze_pretrained is not None:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        assert length == self.seq_length, f"{length} <> {self.seq_length}"
        assert hidden_size == self.d_encoder, f"{hidden_size} <> {self.d_encoder}"

        # compress
        y = self.fc_compress(x)  # -> B x L x H (example: 32 x 512 x 100)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 51_200)
        # first latent var
        z_1 = self.fc_z_1_1(y)  # -> B x Z_1  (example: 32 x (20**2)*10)
        z_1 = self.norm_z_1(z_1)
        z_1 = self.act_fct(z_1)

        y_1 = self.fc_z_1_2(z_1)  # -> B x (L * H)
        y_1 = self.norm_decompress_1(y_1)
        y_1 = self.act_fct(y_1)

        # second latern var
        z_2 = self.fc_z_2_1(y.clone())  # -> B x Z_2  (example: 32 x (20*10)
        z_2 = self.norm_z_2(z_2)
        z_2 = self.act_fct(z_2)

        y_2 = self.fc_z_2_2(z_2)  # -> B x (L * H)
        y_2 = self.norm_decompress_2(y_2)
        y_2 = self.act_fct(y_2)

        # combaine
        y_1 = y_1.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        y_2 = y_2.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        y = torch.cat([y_1, y_2], -1)  # -> B x L x 2*H

        # decompress
        y = self.fc_decompress(y)  # -> B x L x H_dec
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

        # trainable language model head
        lm_logits = self.lm_head(decoder_outputs[0])  # maybe not necessary

        # calculate composite loss
        if labels is not None:
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            with torch.no_grad():
                edges, node_embeddings = sample_graph(self.num_nodes, self.sample_num_entities, self.sample_num_interactions, self.num_node_features, self.sampling_iterations)
            adj_matrix_distro_loss = mmd(edges.view(self.sampling_iterations, self.num_nodes ** 2), z_1)
            node_label_distro_loss = mmd(node_embeddings.view(self.sampling_iterations, self.num_nodes * self.num_node_features), z_2)
            adj_matrix_distro_loss = self.alpha * adj_matrix_distro_loss
            node_label_distro_loss = self.beta * node_label_distro_loss
            loss = masked_lm_loss  + adj_matrix_distro_loss + node_label_distro_loss
            supp_data = torch.cat([
                masked_lm_loss.unsqueeze_(0),
                adj_matrix_distro_loss.unsqueeze_(0),
                node_label_distro_loss.unsqueeze_(0)
            ], 0).unsqueeze(0)
        else:
            loss = None
            supp_data = None

        return BecauseOutput(
            loss=loss,
            logits=lm_logits,
            # supplementarty data for logging only
            supp_data=supp_data,
            z=None,  #z,
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
