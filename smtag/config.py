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
from ray.tune.schedulers import PopulationBasedTraining, pbt
from ray import tune
from ray.tune import CLIReporter

@dataclass
class Config:
    max_length: int = 512  # in tokens!
    truncation: bool = True
    min_char_length: int = 80  # characters!
    split_ratio: InitVar[Dict] = None
    celery_batch_size: int = 1000
    from_pretrained: str = "roberta-base"  # "facebook/bart-base" # leave empty if training a language model from scratch
    model_type: str = "Autoencoder"  # "VAE" #  "Twin"
    nlp: English = field(default=spacy.load("en_core_web_sm"))
    tokenizer: InitVar[Union[RobertaTokenizerFast, BartTokenizerFast, ByT5Tokenizer]] = None
    fast: bool = True
    split_ratio: InitVar[Dict[str, float]] = None
    asynchr: bool = True
    twin_delimiter: str = "###tt9HHSlkWoUM###"  # to split concatenated twin examples
    hp_search_config: dict = None
    hp_search_scheduler: pbt.PopulationBasedTraining = PopulationBasedTraining()
    hp_search_reporter: tune.progress_reporter.CLIReporter = CLIReporter()

    def __post_init__(
        self,
        split_ratio,
        tokenizer
    ):
        self.split_ratio = {
            "train": 0.8,
            "eval": 0.1,
            "test": 0.1,
            "max_eval": 10_000,
            "max_test": 10_000
        } if split_ratio is None else split_ratio
        # a specific tokenizer can be provided for example when training a model from scratch
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(self.from_pretrained, fast=self.fast)


# char_level_tokenizer = AutoTokenizer.from_pretrained("google/canine-c") # "google/byt5-small") #
# config = Config(tokenizer=char_level_tokenizer)

# config for Twin
# config = Config(
#     max_length=[256, 256],  #[64, 512],  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
#     from_pretrained="facebook/bart-base",  # leave empty if training a language model from scratch
#     model_type="Twin",  # "VAE" #  "Twin"  # "Autoencoder"
#     asynchr=True  # we need ordered examples while async returns results in non deterministic way
# )

# config for QandA
# config = Config(
#     max_length=[256, 256],  #[64, 512],  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
#     from_pretrained= "facebook/bart-base", #"facebook/opt-1.3b", #"facebook/bart-base", # t5-base  # leave empty if training a language model from scratch
#     # fast=False, # for OPT model
#     model_type="Autoencoder",
#     asynchr=True  # we need ordered examples while async returns results in non deterministic way
# )

# config for AndQ
# config = Config(
#     max_length=[256, 256],  #[64, 512],  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
#     from_pretrained="facebook/bart-base", # t5-base  # leave empty if training a language model from scratch
#     model_type="Autoencoder",
#     asynchr=True  # we need ordered examples while async returns results in non deterministic way
# )

# config for VAE

# config = Config(
#     max_length=512, #[64, 512],  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
#     from_pretrained="facebook/bart-base",  # leave empty if training a language model from scratch
#     model_type="VAE",  # "VAE" #  "Twin"  # "Autoencoder"
#     asynchr=True  # we need ordered examples while async returns results in non deterministic way
# )

# config for nomral language model
# config = Config(
#     max_length=512,  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
#     from_pretrained="roberta-base",  # leave empty if training a language model from scratch
#     model_type="Autoencoder",
#     asynchr=True  # we need ordered examples while async returns results in non deterministic way
# )

# # Default config for token classification / Roberta
# config = Config()


# Configuration for hyperparameter tuning
#####################################################################
HP_SEARCH_CONFIG = {
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32]),
        "per_device_eval_batch_size": 64,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        # "lr_scheduler": tune.choice(["cosine", "linear", "constant"]),
        # "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        "adam_beta1": tune.choice([0.75, 0.8, 0.85, 0.9, 0.95]),
        # "adam_beta2": tune.uniform(0.950, 0.999),
        "adam_epsilon": tune.loguniform(1e-10, 1e-6),
   }

HP_SEARCH_SCHEDULER = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.15),
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "per_device_train_batch_size": [4, 8, 16, 32],
            "adam_beta1": [0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
            # "adam_beta2": tune.uniform(0.90, 0.99),
            "adam_epsilon": tune.loguniform(1e-10, 1e-6),
            "adafactor": [True, False],
        },
    )

HP_SEARCH_REPORTER = CLIReporter(
    parameter_columns={
        "weight_decay": "w_decay",
        "learning_rate": "lr",
        # "lr_scheduler": "lr_schedule",
        "per_device_train_batch_size": "train_bs/gpu",
        "adam_beta1": "adam_beta1",
        # "adam_beta2": "adam_beta2",
        "adam_epsilon": "adam_epsilon",
        "adafactor": "adafactor",
    },
    metric_columns=["eval_accuracy_score", "eval_precision", "eval_recall", "eval_f1", "epoch", "eval_runtime"],
)

config = Config(
    max_length=512,  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
    from_pretrained="roberta-base",  # leave empty if training a language model from scratch
    model_type="Autoencoder",
    asynchr=True,  # we need ordered examples while async returns results in non deterministic way
    hp_search_config=HP_SEARCH_CONFIG,
    hp_search_scheduler=HP_SEARCH_SCHEDULER,
    hp_search_reporter=HP_SEARCH_REPORTER
)

