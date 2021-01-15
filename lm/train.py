# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
# Just for reference... 
# Roberta:
# We consider five English-language corpora of varying sizes and domains, 
# totaling over 160GB of uncompressed text.
# The model was trained on 1024 V100 GPUs for 500K steps with a batch size of 8K 
# and a sequence length of 512. 
# The optimizer used is Adam with a learning rate of 6e-4, 
# \beta_{1} = 0.9  \beta_{2} = 0.98β and \epsilon = 1e-6
# a weight decay of 0.01, learning rate warmup for 24,000 steps 
# and linear decay of the learning rate after.

from typing import NamedTuple
from pathlib import Path
import torch
from dataclasses import dataclass, field
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    TrainingArguments, HfArgumentParser,
    DataCollatorForLanguageModeling, EvaluationStrategy
)
from datasets import load_dataset, GenerateMode
from .trainer import MyTrainer
from .data_collator import DataCollatorForPOSMaskedLanguageModeling
from .show import ShowExample
from .metrics import compute_metrics

from common.config import config
from common import TOKENIZER_PATH, LM_DATASET, LM_MODEL_PATH, CACHE


def train(no_cache: bool, dataset_path: str, data_config_name: str, training_args: TrainingArguments):
    print(f"Loading tokenizer from {TOKENIZER_PATH}.")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        TOKENIZER_PATH,
        # max_len=config.max_length
    )

    print(f"\nLoading and tokenizing datasets found in {dataset_path}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './lm/loader.py',
        data_config_name,
        data_dir=dataset_path,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=CACHE,
        tokenizer=tokenizer
    )
    if data_config_name != "MLM":
        data_collator = DataCollatorForPOSMaskedLanguageModeling(
            tokenizer=tokenizer,
            max_length=config.max_length
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True
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
        compute_metrics=compute_metrics,
        callbacks=[ShowExample(tokenizer)]
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    trainer.train()
    trainer.save_model(training_args.output_dir)

    print(f"Testing on {len(test_dataset)}.")
    pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
    print(f"{pred.metrics}")


if __name__ == "__main__":

    @dataclass
    class MyTrainingArguments(TrainingArguments):
        output_dir: str = field(default=LM_MODEL_PATH)
        overwrite_output_dir: bool = field(default=True)
        logging_steps: int = field(default=50)

    parser = HfArgumentParser((MyTrainingArguments), description="Traing script.")
    parser.add_argument("dataset_path", nargs="?", default=LM_DATASET, help="The dataset to use for training.")
    parser.add_argument("data_config_name", nargs="?", default="MLM", choices=["MLM", "DET", "VERB", "SMALL"], help="Name of the dataset configuration to use.")
    parser.add_argument("--no-cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cacher.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    dataset_path = args.dataset_path
    data_config_name = args.data_config_name
    if Path(training_args.output_dir).exists():
        train(no_cache, dataset_path, data_config_name, training_args)
    else:
        print(f"{training_args.output_dir} does not exist! Cannot proceed.")
