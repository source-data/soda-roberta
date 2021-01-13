# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
# Just for reference... 
# Roberta:
# The model was trained on 1024 V100 GPUs for 500K steps with a batch size of 8K 
# and a sequence length of 512. 
# The optimizer used is Adam with a learning rate of 6e-4, 
# \beta_{1} = 0.9  \beta_{2} = 0.98β and \epsilon = 1e-6
# a weight decay of 0.01, learning rate warmup for 24,000 steps 
# and linear decay of the learning rate after.

from typing import NamedTuple
import torch
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    TrainingArguments, HfArgumentParser,
    DataCollatorForLanguageModeling, EvaluationStrategy
)
from datasets import load_dataset, GenerateMode
from .trainer import MyTrainer
from .data_collator import DataCollatorForPOSMaskedLanguageModeling
from .metrics import compute_metrics

from common.config import config
from common import TOKENIZER_PATH, LM_DATASET, HUGGINGFACE_CACHE


def train(no_cache: bool, data_config_name: str, training_args: TrainingArguments, mlm_probability: float):
    print(f"Loading tokenizer from {TOKENIZER_PATH}.")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        TOKENIZER_PATH,
        # max_len=config.max_length
    )

    print(f"\nLoading and tokenizing datasets found in {LM_DATASET}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './lm/loader.py',
        data_config_name,
        data_dir=LM_DATASET,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=HUGGINGFACE_CACHE,
        tokenizer=tokenizer
    )
    if data_config_name == "MASKED_DET":
        data_collator = DataCollatorForPOSMaskedLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            padding=True,
            max_length=config.max_length
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")

    model_config = RobertaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_length + 2,  # max_length + 2 for start/end token
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=model_config)
    training_args.remove_unused_columns = False
    training_args.evaluation_strategy = EvaluationStrategy.STEPS
    print("\nTraining arguments:")
    print(training_args)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    trainer.train()
    trainer.save_model(training_args.output_dir)

    print(f"Testing on {len(test_dataset)}.")
    pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
    print(f"{pred.metrics}")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments), description="Traing script.")
    parser.add_argument("data_config_name", nargs="?", default="MLM", choices=["MLM", "MASKED_DET"], help="Name of the dataset configuration to use.")
    parser.add_argument("--no-cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cacher.")
    parser.add_argument("--mlm_probability", default=1.0, type=float, help="Probability of masking.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    data_config_name = args.data_config_name
    mlm_probability = args.mlm_probability
    train(no_cache, data_config_name, training_args, mlm_probability)
