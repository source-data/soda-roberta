"""
Application-wide preferences.
"""
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import spacy
from spacy.lang.en import English
from typing import Dict
from . import TOKENIZER_PATH


@dataclass
class Config:
    vocab_siz: int = 54_000
    max_length: int = 64  # 512  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
    truncation: bool = True
    min_char_length: int = 120  # characters!
    split_ratio: Dict = field(default_factory=dict)
    celery_batch_size: int = 1000
    from_pretrained: str = "facebook/bart-base" # "roberta-base"  #    # leave empty if training a language model from scratch
    model_type: str = "GraphRepresentation" # "Autoencoder"  # 
    tokenizer: str = None
    nlp: English = field(default=spacy.load("en_core_web_sm"))

    def __post_init__(
        self,
        split_ratio: Dict = {
            "train": 0.7,
            "eval": 0.2,
            "test": 0.1,
            "max_eval": 10_000,
            "max_test": 10_000}):
        self.split_ratio = split_ratio


config = Config()

if config.from_pretrained:
    config.tokenizer = AutoTokenizer.from_pretrained(config.from_pretrained)
else:
    config.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)