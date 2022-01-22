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
from typing import List, Dict, Optional, Tuple, Union, Any
import torch

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import (
    PaddingStrategy, DataCollatorMixin,
    _torch_collate_batch
)


"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""


# @dataclass
# class DataCollatorForTargetedMasking():
#     """
#     Data collator used for random masking of targeted classes of token.
#     Useful for learning language models based on masking part-of-speech tokens. 
#     Instead of masking any random token as in MLM, only token that belong to a defined class are masked.
#     Inputs, labels and masks are dynamically padded to the maximum length of a batch if they are not all of the same length.

#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         mlm_probability (:obj:`float`, `optional`, defaults to 1.0):
#             The probability with which to mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
#     .. note::

#         This data collator expects a dataset having items that are dictionaries
#         with the "special_tokens_mask" and "pos_mask" keys.
#     """

#     tokenizer: PreTrainedTokenizerBase
#     mlm_probability: float = 1.0
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None

#     def __post_init__(self):
#         if self.tokenizer.mask_token_id is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#             )

#     def call(self, features) -> Dict[str, torch.Tensor]:
#         """
#         In addition to input_ids, a feature 'tag_mask' needs to be provided to specify which token might be masked.
#         """
#         tag_mask = [feature['tag_mask'] for feature in features] if 'tag_mask' in features[0].keys() else None
#         if tag_mask is None:
#             raise ValueError(
#                 "A mask should be provided to indicate which input token class to mask."
#             )
#         if self.pad_to_fixed_length and not self.pad_to_multiple_of:
#             import pdb; pdb.set_trace()
#             batch = self.tokenizer.pad(
#                 features,
#                 return_tensors="pt",
#                 padding='max_length',
#                 max_length=self.pad_to_fixed_length
#             )
#         elif self.pad_to_multiple_of and not self.pad_to_fixed_length:
#             batch = self.tokenizer.pad(
#                 features,
#                 padding=self.padding,
#                 max_length=self.max_length,
#                 pad_to_multiple_of=self.pad_to_multiple_of
#             )
#         elif self.pad_to_fixed_length and self.pad_to_multiple_of:
#             raise ValueError('pad_to_fixed_length and pad_to_multiple_of are mutually exclusive options.')

#         batch['tag_mask'] = tag_mask
#         sequence_length = len(batch["input_ids"][0])
#         padding_side = self.tokenizer.padding_side
#         if padding_side == "right":
#             batch["tag_mask"] = [
#                 x + [0] * (sequence_length - len(x)) for x in batch["tag_mask"]
#             ]
#         else:
#             batch["tag_mask"] = [
#                 [0] * (sequence_length - len(x)) + x for x in batch["tag_mask"]
#             ]
#         batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
#         batch["input_ids"], batch["labels"] = self.tag_mask_tokens(batch["input_ids"], batch["tag_mask"])
#         batch.pop("tag_mask")
#         return batch

#     def tag_mask_tokens(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Masks the input as specified by the tag mask prepared by the loader"""
#         targets = inputs.clone()
#         # create and initialize the probability matric for masking to zeros
#         probability_matrix = torch.zeros_like(targets, dtype=torch.float64)
#         # update in-place probability to the set mlm_probability value where mask is true
#         probability_matrix.masked_fill_(mask.bool(), value=self.mlm_probability)
#         # use the probability at each position to randomly mask or not
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         # reolace input_ids by the mask token id at position that need to be masked
#         inputs[masked_indices] = self.tokenizer.mask_token_id
#         # we train to only predict the masked position
#         targets[~masked_indices] = -100  # We only compute loss on masked tokens
#         return inputs, targets


# @dataclass
# class DataCollatorForLanguageModelingWithFixedLength(DataCollatorMixin):
#     """
#     Data collator used for language modeling. Inputs are padded to the fixed length
#     so that they are all of the the same length.
#     Args:
#         tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
#             The tokenizer used for encoding the data.
#         mlm (`bool`, *optional*, defaults to `True`):
#             Whether or not to use masked language modeling. If set to `False`, the labels are the same as the
#             inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
#             non-masked tokens and the value to predict for the masked token.
#         mlm_probability (`float`, *optional*, defaults to 0.15):
#             The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
#         pad_to_fixed_length (`int`, *optional*, defaults to 1024):
#             If set will pad the sequence to the provided value.
#         return_tensors (`str`):
#             The type of Tensor to return. Allowable values are "np", "pt" and "tf".
#     <Tip>
#     For best performance, this data collator should be used with a dataset having items that are dictionaries or
#     BatchEncoding, with the `"special_tokens_mask"` key, as returned by a
#     [`PreTrainedTokenizer`] or a [`PreTrainedTokenizerFast`] with the
#     argument `return_special_tokens_mask=True`.
#     </Tip>"""

#     tokenizer: PreTrainedTokenizerBase
#     mlm: bool = True
#     mlm_probability: float = 0.15
#     pad_to_fixed_length: Optional[int] = 1024
#     tf_experimental_compile: bool = False
#     return_tensors: str = "pt"

#     def __post_init__(self):
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )
#         if self.tf_experimental_compile:
#             import tensorflow as tf

#             self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

#     def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
#         # Handle dict or lists with proper padding and conversion to tensor.
#         # if isinstance(examples[0], (dict, BatchEncoding)):
#         #     batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
#         # else:
#         #     batch = {
#         #         "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
#         #     }

#         import pdb; pdb.set_trace()
#         batch = self.tokenizer.pad(
#             examples,
#             return_tensors="pt",
#             padding="max_length",
#             max_length=self.pad_to_fixed_length
#         )

#         # If special token mask has been preprocessed, pop it from the dict.
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
#         if self.mlm:
#             batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
#                 batch["input_ids"], special_tokens_mask=special_tokens_mask
#             )
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch

#     def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
#         """
#         Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#         """
#         import torch

#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#         probability_matrix = torch.full(labels.shape, self.mlm_probability)
#         if special_tokens_mask is None:
#             special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#             ]
#             special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
#         else:
#             special_tokens_mask = special_tokens_mask.bool()

#         probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels


@dataclass
class DataCollatorForTargetedMasking(DataCollatorMixin):
    """
    Data collator used for random masking of targeted classes of token.
    Useful for learning language models based on masking part-of-speech tokens. 
    Instead of masking any random token as in MLM, only token that belong to a defined class are masked.
    Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 1.0
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # In addition to input_ids, a feature 'tag_mask' needs to be provided to specify which token might be masked.
        tag_mask = [example['tag_mask'] for example in examples] if 'tag_mask' in examples[0].keys() else None
        if tag_mask is None:
            raise ValueError(
                "A mask should be provided to indicate which input token class to mask."
            )
        # pop tag_mask from examples before padding to avoid tokenizer being confused
        for e in examples:
            e.pop('tag_mask')
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        # special_tokens_mask = batch.pop("special_tokens_mask", None)

        # padding the mask (not handled by the tokenizer) to same uniform length as input_ids
        sequence_length = len(batch["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            tag_mask = [x + [0] * (sequence_length - len(x)) for x in tag_mask]
        else:
            tag_mask = [[0] * (sequence_length - len(x)) + x for x in tag_mask]
        # tensorify the mask
        tag_mask = torch.tensor(tag_mask, dtype=torch.uint8)
        # input_ids are already tensors, see tokenizer return_tensors="pt"
        batch["input_ids"], batch["labels"] = self.torch_tag_mask_tokens(batch["input_ids"], tag_mask)
        return batch

    def torch_tag_mask_tokens(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masks the input as specified by the tag mask prepared by the loader"""
        targets = inputs.clone()
        # create and initialize to zeros the probability matrix for masking
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
