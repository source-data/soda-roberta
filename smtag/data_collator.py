
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

from termios import PENDIN
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Any
import torch

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import (
    DataCollatorMixin, DataCollatorForLanguageModeling,
    _torch_collate_batch
)
from transformers.file_utils import PaddingStrategy


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
        # in case labels are provided by a token classification dataset, pop them too
        for e in examples:
            e.pop('tag_mask')
            if 'labels' in e:
                e.pop('labels')
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

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


@dataclass
class DataCollatorForMaskedTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        select_labels (bool, defaults to False):
            Whether use only the labels at the masked position to calculate the loss
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    masking_probability: float = .0
    replacement_probability: float = .0
    select_labels: bool = False

    def torch_call(self, features) -> Dict[str, torch.Tensor]:
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
            batch["labels"] = [x + [self.label_pad_token_id] * (sequence_length - len(x)) for x in labels]
        else:
            batch["tag_mask"] = [[0] * (sequence_length - len(x)) + x for x in tag_mask]
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(x)) + x for x in labels]
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


@dataclass
class MyDataCollatorForSeq2Seq(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            if self.max_length is not None:
                assert max_label_length <= self.max_length, f"{max_label_length} > {self.max_length}"

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class MyDataCollatorForTwinSeq2Seq(MyDataCollatorForSeq2Seq):
    """
    Data collator for inputs that are provided as lists of twin examples.
    It will dynamically pad the inputs received, as well as the labels.
    """

    max_length_list: List[int] = None # each twin example has its max length!

    def __call__(self, features, return_tensors=None):
        num_twins = len(features[0]["input_ids"])
        appended_features = []
        for twin_idx in range(num_twins):
            # extract a specific twin example from the features
            features_this_twin = [
                {
                    k: v[twin_idx] for k, v in feature.items()
                }
                for feature in features
            ]
            self.pad_to_multiple_of = self.max_length_list[twin_idx]
            appended_features.append(super().__call__(features_this_twin, return_tensors))
        # from list of dict to dict of list
        new_features = {k: [] for k in appended_features[0].keys()}
        for f in appended_features:
            for k, v in f.items():
                new_features[k].append(v)
        return new_features


@dataclass
class MyDataCollatorForTwinLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator for inputs that are provided as lists of twin examples.
    It will dynamically pad the inputs received, as well as the labels.
    """

    max_length_list: List[int] = None # each twin example has its max length!

    def __call__(self, features, return_tensors=None):
        num_twins = len(features[0]["input_ids"])
        appended_features = []
        for twin_idx in range(num_twins):
            # extract a specific twin example from the features
            features_this_twin = [
                {
                    k: v[twin_idx] for k, v in feature.items()
                }
                for feature in features
            ]
            self.pad_to_multiple_of = self.max_length_list[twin_idx]
            appended_features.append(super().__call__(features_this_twin, return_tensors))
        # from list of dict to dict of list
        new_features = {k: [] for k in appended_features[0].keys()}
        for f in appended_features:
            for k, v in f.items():
                new_features[k].append(v)
        # stack list into tensor
        # for k, v in new_features.items():
        #     new_features[k] = torch.stack(v, dim=-1)
        return new_features
