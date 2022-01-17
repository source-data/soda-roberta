from itertools import product
from random import sample, gauss
from dataclasses import dataclass
import torch
from torch import nn
from transformers import (
    BartConfig, BartTokenizerFast, BartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, BaseModelOutputWithPastAndCrossAttentions


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
    def __init__(self, max_nodes=None, num_features=None, *args, **kwargs):
        super(BecauseConfig).__init__(*args, **kwargs)
        self.max_nodes = max_nodes
        self.num_features = num_features


@dataclass
class BecauseOutput(MaskedLMOutput):
    z_1: torch.Tensor = None
    z_2: torch.Tensor = None


class Because(nn.Module):

    def __init__(
        self,
        pretrained: BartForConditionalGeneration,
        max_nodes=None,
        num_features=None,
        num_entities=3,
        num_interactions=3,
        sampling_iterations=10
    ):
        super(Because, self).__init__()#config=pretrained.config, *args, **kwargs)
        # from the pretrained model
        self.pretrained = pretrained
        self.encoder = self.pretrained.get_encoder()
        self.decoder = self.pretrained.get_decoder()  # TODO: check it uses cross attention?
        self.d_encoder = self.encoder.config.d_model
        self.d_decoder = self.decoder.config.d_model
        self.pad_token_id = self.decoder.config.pad_token_id
        self.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        self.lm_head = self.pretrained.lm_head
        # latent vars
        self.max_nodes = max_nodes
        self.num_features = num_features
        self.num_entities = num_entities
        self.num_interactions = num_interactions
        self.sampling_iterations = sampling_iterations
        self.z1_dim = self.max_nodes ** 2
        self.z2_dim = self.max_nodes * self.num_features
        # own layers
        self.dropout = nn.Dropout(0.1)
        self.act_fn = nn.GELU()
        self.fc_enc_1 = nn.Linear(self.d_encoder, self.d_encoder)
        self.norm_enc = nn.LayerNorm(self.d_encoder)
        self.fc_z1 = nn.Linear(self.d_encoder, self.z1_dim)
        self.fc_z2 = nn.Linear(self.d_encoder, self.z2_dim)
        self.fc_dec_1 = nn.Linear(self.z1_dim, self.d_decoder)
        self.fc_dec_2 = nn.Linear(self.z2_dim, self.d_decoder)
        self.conv1d = nn.Conv1d(2, 1, 1, 1)
        self.norm_dec = nn.LayerNorm(self.d_decoder)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None, **kwargs) -> BecauseOutput:
        # encode
        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)
        y: BaseModelOutput = encoder_outputs[0]
        # encoder output to latent variable
        y = self.dropout(y)
        y = self.fc_enc_1(y)
        y = self.act_fn(y)
        z = self.norm_enc(y)
        # latent variable to adj matrix
        z_1 = self.fc_z1(z).view(-1, self.max_nodes * self.max_nodes)
        # latent variable to entity embeddings
        z_2 = self.fc_z2(z).view(-1, self.max_nodes * self.num_features)
        # adj matrix and entity embeddings to decoder input
        y_1 = self.fc_dec_1(z_1)
        y_2 = self.fc_dec_2(z_2)
        # introduce third dimension to be able to concatenate vectors
        y_1 = y_1.unsqueeze(1)
        y_2 = y_2.unsqueeze(1)
        y = torch.cat([y_1, y_2], 1)
        y = self.dropout(y)
        y = self.conv1d(y)
        y = self.act_fn(y)
        # remove extra dimension again after having reduced the concatenated vectors
        y = y.squeeze(1)
        y = self.norm_dec(y)
        # decode
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
        lm_logits = self.lm_head(decoder_outputs[0])  # maybe not necessary
        # calculate loss
        if labels is not None:
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            with torch.no_grad():
                adj_sampling, node_label_sampling = interaction_samples(self.max_nodes, self.num_entities, self.num_interactions, self.num_features, self.sampling_iterations)
            adj_matrix_distro_loss = mmd(adj_sampling.view(self.sampling_iterations, self.max_nodes ** 2), z_1)
            node_label_distro_lost = mmd(node_label_sampling.view(self.sampling_iterations, self.max_nodes * self.num_features), z_2)
            loss = masked_lm_loss + adj_matrix_distro_loss + node_label_distro_lost
        else:
            loss = None
        # return both latent representations and output
        # The Trainer class is optimized for 🤗 Transformers models and can have surprising behaviors when you use it on other models. When using it on your own model, make sure:

        # your model always return tuples or subclasses of ModelOutput.
        # your model can compute the loss if a labels argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
        # your model can accept multiple label arguments (use the label_names in your TrainingArguments to indicate their name to the Trainer) but none of them should be named "label".
        return BecauseOutput(
            loss=loss,
            logits=lm_logits,
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
    seq2seq = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    because = Because(pretrained=seq2seq, max_nodes=4, num_features=10, sampling_iterations=10)
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    test_string = "Huggingface office is based in "
    inputs = tokenizer([test_string] * 32, return_tensors="pt")
    input_ids = inputs["input_ids"]
    labels = input_ids.masked_fill_(input_ids == seq2seq.config.pad_token_id, -100)
    outputs = because(input_ids=input_ids, labels=labels)
    output_ids = torch.argmax(torch.softmax(outputs.logits, -1), -1)
    result_string = tokenizer.decode(output_ids[0].tolist())
    print(result_string)


def main():
    self_test()


if __name__ == "__main__":
    self_test()
