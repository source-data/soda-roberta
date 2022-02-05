# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from transformers.data.data_collator import PaddingStrategy
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""


@dataclass
class DataCollatorForMaskedTokenClassification:
    """
    Data collator used for random masking of targeted classes of tokens.
    Useful for enforcing learning from context: if the token to be classified is masked, only the context can provide information. 
    Instead of masking any random token as in masked language modeling, only the token that belong to a defined class are masked.
    Inputs, labels and masks are dynamically padded to the maximum length of a batch if they are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        masking_probability (:obj:`float`, `optional`, defaults to 0):
            The probability with which to mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        select_labels (bool, defaults to False): whether use only the labels at the masked position to calculate the loss

    .. note::

        This data collator expects a dataset having items that are dictionaries
        with the "special_tokens_mask" and "pos_mask" keys.
    """

    tokenizer: PreTrainedTokenizerBase
    masking_probability: float = .0
    replacement_probability: float = .0
    padding: Union[bool, str, PaddingStrategy] = True
    pad_token_id: int = -100
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    select_labels: bool = False

    def __post_init__(self):
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked token classification. "
            )
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "This tokenizer does not have a padding token which is necessary for masked token classification. "
            )

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        """
        In addition to input_ids, a feature 'tag_mask' needs to be provided to specify which token might be masked.
        """
        if 'tag_mask' in features[0].keys():
            tag_mask = [feature['tag_mask'] for feature in features]
        else:
            raise ValueError("A mask should be provided to indicate which input token class to mask.")
        label_name = "label" if "label" in features[0].keys() else "labels"
        if label_name in features[0].keys():
            labels = [feature[label_name] for feature in features]
        else:
            raise ValueError("A feature 'label' or 'labels' should be provided for token classification")
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        # batch['input_ids'] are now padded
        # we still need to 'manually' pad the labels and the tag mask
        sequence_length = len(batch["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["tag_mask"] = [x + [0] * (sequence_length - len(x)) for x in tag_mask]
            batch["labels"] = [x + [self.pad_token_id] * (sequence_length - len(x)) for x in labels]
        else:
            batch["tag_mask"] = [[0] * (sequence_length - len(x)) + x for x in tag_mask]
            batch["labels"] = [[self.pad_token_id] * (sequence_length - len(x)) + x for x in labels]
        # convert dict of list of lists into ditc of tensors
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # stochastically mask input ids according to tag_mask
        batch["input_ids"], batch["labels"] = self.tag_mask_tokens(batch["input_ids"], batch["labels"], batch["tag_mask"])
        # remove tak_mask from match as it would be rejected by model.forward()
        batch.pop("tag_mask")
        return batch

    def tag_mask_tokens(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masks the input as specified by the tag mask prepared by the loader"""
        inputs = inputs.clone()  # not sure if necessary; might be safer to avoid changing input features when provided as tensor
        if self.select_labels:
            targets = targets.clone()
        # create the probability matrix for masking
        masking_probability_matrix = torch.full(inputs.size(), self.masking_probability)
        # use the probability matrix to draw whether to replace or not and intersect with the mask
        masked_indices = torch.bernoulli(masking_probability_matrix).bool() & mask.bool()
        # replace input_ids by the mask token id at position that need to be masked
        inputs[masked_indices] = self.tokenizer.mask_token_id
        # second probability matrix is to determin whether to randomize remaining marked tokens
        replacement_probability_matrix = torch.full(inputs.size(), self.replacement_probability)
        # indices of token to replace found by drawing from prob matric and intersecting with mask but exclusin alreayd masked positions
        replaced_indices = torch.bernoulli(replacement_probability_matrix).bool() & mask.bool() & ~masked_indices
        # draw random int from vocab size of tokenizer and fill tenzoer of shape like intput
        random_input_ids = torch.randint(len(self.tokenizer), inputs.size(), dtype=torch.long)
        # at the replacmenet indices, change to random token
        inputs[replaced_indices] = random_input_ids[replaced_indices]
        if self.select_labels:
            # only labels at the makred position (irrespective of whether they are masked) will be used for calculating the loss
            targets[~mask] = -100
        return inputs, targets
