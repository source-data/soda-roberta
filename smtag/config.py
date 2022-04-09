"""
Application-wide preferences.
"""
from dataclasses import dataclass, field, InitVar
from transformers import (
    AutoTokenizer,
    BartTokenizerFast,
    RobertaTokenizerFast,
    ByT5Tokenizer
)
import spacy
from spacy.lang.en import English
from typing import Dict, Union


@dataclass
class Config:
    max_length: int = 512  # in tokens!
    truncation: bool = True
    min_char_length: int = 120  # characters!
    split_ratio: InitVar[Dict] = None
    celery_batch_size: int = 1000
    from_pretrained: str = "roberta-base"  # "facebook/bart-base" # leave empty if training a language model from scratch
    model_type: str = "Autoencoder"  # "VAE" #  "Twin"
    nlp: English = field(default=spacy.load("en_core_web_sm"))
    tokenizer: InitVar[Union[RobertaTokenizerFast, BartTokenizerFast, ByT5Tokenizer]] = None
    split_ratio: InitVar[Dict[str, float]] = None

    def __post_init__(
        self,
        split_ratio,
        tokenizer
    ):
        self.split_ratio = {
            "train": 0.7,
            "eval": 0.2,
            "test": 0.1,
            "max_eval": 10_000,
            "max_test": 10_000
        } if split_ratio is None else split_ratio
        # a specific tokenizer can be provided for example when training a model from scratch
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(self.from_pretrained)


# char_level_tokenizer = AutoTokenizer.from_pretrained("google/canine-c") # "google/byt5-small") #
# config = Config(tokenizer=char_level_tokenizer)
config = Config(
    max_length=512,  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
    from_pretrained="facebook/bart-base",  # leave empty if training a language model from scratch
    model_type="Twin"  # "VAE" #  "Twin"
)
