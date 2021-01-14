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
from typing import Dict, List, Optional, Tuple, Union
from transformers.data.data_collator import PaddingStrategy
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""


@dataclass
class DataCollatorForPOSMaskedLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        pos_mask = [feature['pos_mask'] for feature in features] if 'pos_mask' in features[0].keys() else None
        special_tokens_mask = [feature['special_tokens_mask'] for feature in features] if 'special_tokens_mask' in features[0].keys() else None

        if pos_mask is None:
            raise ValueError(
                "Part-of-speech masks should be provided to indicate which input token to mask."
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['pos_mask'] = pos_mask
        batch['special_tokens_mask'] = special_tokens_mask
        sequence_length = len(batch["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["pos_mask"] = [
                x + [0] * (sequence_length - len(x)) for x in batch["pos_mask"]
            ]
            batch["special_tokens_mask"] = [
                x + [0] * (sequence_length - len(x)) for x in batch["special_tokens_mask"]
            ]
        else:
            batch["pos_mask"] = [
                [0] * (sequence_length - len(x)) + x for x in batch["pos_mask"]
            ]
            batch["special_tokens_masks"] = [
                [0] * (sequence_length - len(x)) + x for x in batch["special_tokens_mask"]
            ]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["input_ids"], batch["labels"] = self.pos_mask_tokens(batch["input_ids"], batch["pos_mask"])
        batch.pop("pos_mask")
        batch.pop("special_tokens_mask")
        return batch

    def pos_mask_tokens(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masks the input as specified by the mask prepared by the loader"""
        targets = inputs.clone()
        probability_matrix = torch.zeros_like(targets, dtype=torch.float64)
        probability_matrix.masked_fill_(mask, value=self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[masked_indices] = self.tokenizer.mask_token_id
        targets[~masked_indices] = -100  # We only compute loss on masked tokens
        return inputs, targets

