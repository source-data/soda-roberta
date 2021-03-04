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
from datetime import datetime
import torch
from dataclasses import dataclass, field
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    TrainingArguments, HfArgumentParser,
    DataCollatorForLanguageModeling, EvaluationStrategy
)
from datasets import load_dataset, GenerateMode
from .trainer import MyTrainer
from .pos_data_collator import DataCollatorForTargetedMasking
from .show import ShowExample
from .metrics import compute_metrics

from common.config import config
from common import LM_MODEL_PATH, CACHE, RUNS_DIR


def train(
    no_cache: bool,
    dataset_path: str,
    data_config_name: str,
    training_args: TrainingArguments,
    tokenizer: RobertaTokenizerFast
):

    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading datasets found in {dataset_path}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './lm/loader.py',
        data_config_name,
        data_dir=dataset_path,
        split=["train", "validation", "test"],
        # download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=CACHE
    )

    if data_config_name != "MLM":
        data_collator = DataCollatorForTargetedMasking(
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

    if config.from_pretrained:
        model = RobertaForMaskedLM.from_pretrained(config.from_pretrained)
    else:
        model_config = RobertaConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_length + 2,  # max_length + 2 for start/end token
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        model = RobertaForMaskedLM(config=model_config)

    training_args.remove_unused_columns = False  # we need pos_mask and special_tokens_mask in collator

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

    # changing default values
    @dataclass
    class MyTrainingArguments(TrainingArguments):
        output_dir: str = field(default=LM_MODEL_PATH)
        overwrite_output_dir: bool = field(default=True)
        logging_steps: int = field(default=2000)
        evaluation_strategy: EvaluationStrategy = field(default=EvaluationStrategy.STEPS)
        per_device_train_batch_size: int = field(default=16)
        per_device_eval_batch_size: int = field(default=16)
        save_total_limit: int = field(default=5)

    parser = HfArgumentParser((MyTrainingArguments), description="Traing script.")
    parser.add_argument("data_config_name", nargs="?", default="MLM", choices=["MLM", "DET", "VERB", "SMALL"], help="Name of the dataset configuration to use.")
    parser.add_argument("--dataset_path", help="The dataset to use for training.")
    parser.add_argument("--no_cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cacher.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    dataset_path = args.dataset_path
    data_config_name = args.data_config_name
    output_dir_path = Path(training_args.output_dir)
    training_args.logging_dir = f"{RUNS_DIR}/lm-{data_config_name}-{datetime.now().isoformat().replace(':','-')}"
    if not output_dir_path.exists():
        output_dir_path.mkdir()
        print(f"Created {output_dir_path}.")
    tokenizer = config.tokenizer  # tokenizer has to be the same application-wide
    train(no_cache, dataset_path, data_config_name, training_args, tokenizer)
