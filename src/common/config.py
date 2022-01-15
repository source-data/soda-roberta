"""
Application-wide preferences.
"""
from transformers import AutoTokenizer
import spacy
from . import TOKENIZER_PATH


class Config:
    vocab_size = 54_000
    max_length = 512  # tokens!
    min_char_length = 120  # characters
    split_ratio = {
        "train": 0.7,
        "eval": 0.2,
        "test": 0.1,
        "max_eval": 10_000,
        "max_test": 10_000,
    }
    celery_batch_size = 1000
    from_pretrained = "facebook/bart-large"  # "roberta-base"  # leave empty if training a language model from scratch
    model_type = "CausalRepresentation"  # "Autoencoder"
    tokenizer = None
    nlp = spacy.load("en_core_web_sm")


config = Config()

if config.from_pretrained:
    config.tokenizer = AutoTokenizer.from_pretrained(config.from_pretrained, max_len=config.max_length)
else:
    config.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)