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
class DataCollatorForTargetedMasking:
    """
    Data collator used for random masking of targeted classes of token.
    Useful for learning language models based on masking part-of-speech tokens. 
    Instead of masking any random token as in MLM, only token that belong to a defined class are masked.
    Inputs, labels and masks are dynamically padded to the maximum length of a batch if they are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 1.0):
            The probability with which to mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        This data collator expects a dataset having items that are dictionaries
        with the "special_tokens_mask" and "pos_mask" keys.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 1.0
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            )

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        """
        In addition to input_ids, a feature 'tag_mask' needs to be provided to specify which token might be masked.
        """
        tag_mask = [feature['tag_mask'] for feature in features] if 'tag_mask' in features[0].keys() else None
        if tag_mask is None:
            raise ValueError(
                "A mask should be provided to indicate which input token class to mask."
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['tag_mask'] = tag_mask
        sequence_length = len(batch["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["tag_mask"] = [
                x + [0] * (sequence_length - len(x)) for x in batch["tag_mask"]
            ]
        else:
            batch["tag_mask"] = [
                [0] * (sequence_length - len(x)) + x for x in batch["tag_mask"]
            ]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["input_ids"], batch["labels"] = self.tag_mask_tokens(batch["input_ids"], batch["tag_mask"])
        batch.pop("tag_mask")
        return batch

    def tag_mask_tokens(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masks the input as specified by the tag mask prepared by the loader"""
        targets = inputs.clone()
        # create and initialize the probability matric for masking to zeros
        probability_matrix = torch.zeros_like(targets, dtype=torch.float64)
        # update in-place probability to the set mlm_probability value where mask is true
        probability_matrix.masked_fill_(mask.bool(), value=self.mlm_probability)
        # use the probability at each position to randomly mask or not
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # reolace input_ids by the mask token id at position that need to be masked
        inputs[masked_indices] = self.tokenizer.mask_token_id
        # we train to only predict the masked position
        targets[~masked_indices] = -100  # We only compute loss on masked tokens
        return inputs, targets
