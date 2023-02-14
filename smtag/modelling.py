from typing import List, Tuple, Dict, Union, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import ModelOutput
from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util
from allennlp.modules.conditional_random_field import (allowed_transitions,
                                                       ConditionalRandomField)
import numpy as np
VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrfTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    tags: Optional[Tuple[torch.FloatTensor]] = None


class CRFforTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.id2label = config.id2label
        self.constraints = allowed_transitions(
        "BIO", 
        self.id2label
        )
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            num_tags=self.num_labels,
            constraints=self.constraints,
            include_start_end_transitions=True,
        )


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Here is where I need to add the loss from the CRF
            # It is also here where to get the tags
        #   loss_fct = CrossEntropyLoss()
        #   loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            labels_masked = torch.where(labels >= 0, labels, 0)
            tags = torch.zeros_like(labels)
            log_likelihood = self.crf(logits, labels_masked, attention_mask)
            loss = -log_likelihood

        best_paths = self.crf.viterbi_tags(logits, attention_mask, top_k=1)
        for i, path in enumerate(best_paths):
            label_set = path[0][0]
            tags[i][0:len(label_set)] = torch.tensor(label_set)

        # else:
        #     best_paths = self.crf.viterbi_tags(logits, attention_mask, top_k=1)
        #     for i, path in enumerate(best_paths):
        #         label_set = path[0][0]
        #         tags[i][0:len(label_set)] = torch.tensor(label_set)

        # tags = torch.tensor(tags, device=device)
        tags = torch.where(labels != -100, tags, -100)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output + (tags,)) if loss is not None else ((tags,) + output)

        return CrfTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            tags=tags
        )
