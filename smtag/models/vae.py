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
    # x = x.cpu()
    # y = y.cpu()
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    # if torch.cuda.is_available():
    #     mmd = mmd.cuda()
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
    v = torch.zeros(max_num_nodes, node_features)
    p = torch.full_like(v[node_subset], 0.5)
    sample = torch.bernoulli(p)
    v[node_subset] = sample
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

    keys_to_ignore_at_inference = ['adjascency', 'node_embeddings', 'supp_data']

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
        residuals: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.freeze_pretrained = freeze_pretrained
        self.hidden_features = hidden_features
        self.sample_num_entities = sample_num_entities
        self.sample_num_interactions = sample_num_interactions
        self.sample_num_interaction_types = sample_num_interaction_types
        self.sampling_iterations = sampling_iterations
        self.seq_length = seq_length
        self.alpha = alpha
        self.beta = beta
        self.residuals = residuals


class BecauseConfigForTokenClassification(BecauseConfig):

    def __init__(self, classifier_dropout: float = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_dropout = classifier_dropout


@dataclass
class BecauseOutput(MaskedLMOutput):
    supp_data: Dict[str, torch.Tensor] = None
    adjascency: torch.Tensor = None
    node_embeddings: torch.Tensor = None


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

        self.d_encoder = self.encoder.config.d_model
        self.d_decoder = self.decoder.config.d_model
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.decoder.config.pad_token_id
        self.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        self.residuals = self.config.residuals
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

        self.fc_compress = nn.Linear(self.d_encoder, self.hidden_features)
        self.norm_compress = nn.LayerNorm(self.hidden_features, elementwise_affine=False)

        self.fc_z_1_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_1_dim)
        self.norm_z_1 = nn.LayerNorm(self.z_1_dim, elementwise_affine=False)
        self.fc_z_1_2 = nn.Linear(self.z_1_dim, self.seq_length * self.hidden_features)
        self.norm_decompress_1 = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)

        # self.fc_z_2_1 = nn.Linear(self.seq_length * self.hidden_features, self.z_2_dim)
        # self.norm_z_2 = nn.LayerNorm(self.z_2_dim, elementwise_affine=False)
        # self.fc_z_2_2 = nn.Linear(self.z_2_dim, self.seq_length * self.hidden_features)
        # self.norm_decompress_2 = nn.LayerNorm(self.seq_length * self.hidden_features, elementwise_affine=False)

        # self.fc_decompress = nn.Linear(2 * self.hidden_features, self.d_decoder)
        self.fc_decompress = nn.Linear(self.hidden_features, self.d_decoder)

    def forward(self, input_ids=None, labels=None, **kwargs) -> BecauseOutput:
        # encoder
        encoder_outputs: BaseModelOutput = self.encoder(input_ids=input_ids, **kwargs)
        x = encoder_outputs[0]  # B x L x H_enc
        if self.freeze_pretrained in ['encoder', 'both']:
            x.requires_grad_(True)
        batch_size, length, hidden_size = x.size()  # batch_size B, length L, hidden_size H_enc
        # compress
        y = self.fc_compress(x)  # -> B x L x H (example: 32 x 512 x 100)
        y = self.norm_compress(y)
        y = self.act_fct(y)
        y = y.view(batch_size, (self.seq_length * self.hidden_features))  # B x (L * H)  (example: 32 * 51_200)
        # first latent var
        z_1 = self.fc_z_1_1(y)  # -> B x Z_1
        z_1 = self.norm_z_1(z_1)
        z_1 = self.act_fct(z_1)
        # second latent var
        # z_2 = self.fc_z_2_1(y.clone())  # -> B x Z_2  (example: 32 x (20*10)
        # z_2 = self.norm_z_2(z_2)
        # z_2 = self.act_fct(z_2)


        ###########
        z_2 = z_1 #
        ###########


        # decompress
        y_1 = self.fc_z_1_2(z_1)  # -> B x (L * H)
        y_1 = self.norm_decompress_1(y_1)
        y_1 = self.act_fct(y_1)

        # y_2 = self.fc_z_2_2(z_2)  # -> B x (L * H)
        # y_2 = self.norm_decompress_2(y_2)
        # y_2 = self.act_fct(y_2)
        # combine
        y_1 = y_1.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        # y_2 = y_2.view(batch_size, self.seq_length, self.hidden_features)  # -> B x L x H
        # y = torch.cat([y_1, y_2], -1)  # -> B x L x 2*H

        #########
        y = y_1 #
        #########

        # decompress
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

        loss, supp_data = self.compute_loss_on_latent_var(z_1, z_2)

        return BecauseOutput(
            loss=loss,
            logits=logits,
            supp_data=supp_data,
            adjascency=z_1.view(-1, self.num_nodes, self.num_nodes),
            node_embeddings=z_2.view(-1, self.num_nodes, self.num_node_features),
        )

    def compute_loss_on_latent_var(self, z_1, z_2):
        batch_size = z_1.size(0)
        with torch.no_grad():
            edges, node_embeddings = sample_graph(self.num_nodes, self.sample_num_entities, self.sample_num_interactions, self.num_node_features, self.sampling_iterations)
        adj_matrix_distro_loss = self.alpha * mmd(edges.view(self.sampling_iterations, self.num_nodes ** 2), z_1)
        # adj_matrix_distro_loss = torch.zeros(batch_size) # 
        # adj_matrix_distro_loss.requires_grad = True
        # adj_matrix_distro_loss = adj_matrix_distro_loss.cuda()
        ############
        node_label_distro_loss = adj_matrix_distro_loss  # self.beta * mmd(node_embeddings.view(self.sampling_iterations, self.num_nodes * self.num_node_features), z_2)
        ############

        L_adj_sparse = z_1.abs().mean()
        L_node_sparse = z_2.abs().mean()
        # https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
        # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
        # naive (me...):
        d = self.num_nodes  # cosmetic
        W = z_1.view(batch_size, d, d)
        I = torch.eye(d).unsqueeze(0).expand(batch_size, d, d)
        if torch.cuda.is_available():
            W = W.cuda()
            I = I.cuda()
        mat_power_d = torch.matrix_power(I + (W * W ) / d, d)  # based on below Yu et al
        trace = mat_power_d.diagonal(dim1=-1, dim2=-2).sum(-1)
        L_dag = (trace - d).mean()
        #
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
        loss = adj_matrix_distro_loss  # L_dag + node_label_distro_loss + L_adj_sparse + L_node_sparse
        supp_data = {
            "adj_distro_loss": adj_matrix_distro_loss,
            "nodes_distro_loss": node_label_distro_loss,
            "L_adj_sparse": L_adj_sparse,
            "L_dag": L_dag,
            "L_node_sparse": L_node_sparse,
        }
        return loss, supp_data


class BecauseForMaskedLM(Because):

    def __init__(self, pretrained, config: BecauseConfig, **kwargs):
        super().__init__(pretrained, config)
        self.model = Because(pretrained, config)
        self.lm_head = nn.Linear(self.pretrained.config.d_model, self.pretrained.model.shared.num_embeddings, bias=False)

    def forward(self, input_ids, labels,  **kwargs) -> BecauseOutput:
        outputs = self.model(input_ids, labels, **kwargs)
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
        return BecauseOutput(
            loss=loss,
            logits=logits,
            supp_data=supp_data,
            adjascency=outputs.adjascency,
            node_embeddings=outputs.node_embeddings
        )


class BecauseTokenClassification(Because):

    def __init__(self, pretrained, config: BecauseConfigForTokenClassification):
        super().__init__(pretrained, config)
        self.num_labels = config.num_labels
        self.model = Because(pretrained, config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.d_decoder, config.num_labels)

    def forward(self, input_ids, labels, **kwargs) -> BecauseOutput:
        outputs = self.model(input_ids, labels, **kwargs)
        supp_data = outputs.supp_data
        sequence_output = outputs.logits
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
         # calculate composite loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_tokcl = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = outputs.loss + loss_tokcl
            supp_data['loss_tokcl'] = loss_tokcl
        else:
            loss = None
            supp_data['loss_tokcl'] = None

        return BecauseOutput(
            loss=loss,
            logits=logits,
            supp_data=supp_data,
            adjascency=outputs.adjascency,
            node_embeddings=outputs.node_embeddings
        )
