# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass, field
import torch
from transformers import (
    RobertaForTokenClassification, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, HfArgumentParser, EvaluationStrategy
)
from datasets import load_dataset, GenerateMode
# from datasets.utils.download_manager import GenerateMode
from .metrics import MetricsComputer
from .show import ShowExample
from common.config import config
from common import LM_MODEL_PATH, TOKENIZER_PATH, TOKCL_DATASET, TOKCL_MODEL_PATH, CACHE


def train(no_cache: bool, dataset_path: str, data_config_name: str, training_args: TrainingArguments, tokenizer: RobertaTokenizerFast):
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading and tokenizing datasets found in {dataset_path}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './tokcl/loader.py',
        data_config_name,
        data_dir=dataset_path,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=CACHE,
        tokenizer=tokenizer
    )
    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        max_length=config.max_length
    )

    num_labels = train_dataset.info.features['labels'].feature.num_classes
    label_list = train_dataset.info.features['labels'].feature.names
    print(f"\nTraining on {num_labels} features:")
    print(", ".join(label_list))

    compute_metrics = MetricsComputer(label_list=label_list)

    if config.from_pretrained:
        model = RobertaForTokenClassification.from_pretrained(config.from_pretrained, num_labels=num_labels)
    else:
        model = RobertaForTokenClassification.from_pretrained(LM_MODEL_PATH, num_labels=num_labels)

    print("\nTraining arguments:")
    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ShowExample(tokenizer, label_list)]
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
        output_dir: str = field(default=TOKCL_MODEL_PATH)
        overwrite_output_dir: bool = field(default=True)
        logging_steps: int = field(default=50)
        evaluation_strategy: EvaluationStrategy = EvaluationStrategy.STEPS

    parser = HfArgumentParser((TrainingArguments), description="Traing script.")
    parser.add_argument("dataset_path", nargs="?", default=TOKCL_DATASET, help="The dataset to use for training.")
    parser.add_argument("data_config_name", nargs="?", default="NER", choices=["NER", "ROLES", "BORING", "PANELIZATION", "CELL_TYPE_LINE", "GENEPROD"], help="Name of the dataset configuration to use.")
    parser.add_argument("--no_cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cacher.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    data_config_name = args.data_config_name
    dataset_path = args.dataset_path
    output_dir_path = Path(training_args.output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()
        print(f"Created {output_dir_path}.")
    if config.from_pretrained:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', max_len=config.max_length)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)
    train(no_cache, dataset_path, data_config_name, training_args, tokenizer)
