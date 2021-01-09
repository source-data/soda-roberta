# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from argparse import ArgumentParser
from pathlib import Path
import torch
from transformers import (
    RobertaForTokenClassification, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer
)
from datasets import load_dataset, GenerateMode
# from datasets.utils.download_manager import GenerateMode
from .metrics import MetricsComputer
from .show import ShowExample
from common.config import config
from common import tokenizer_PATH, NER_DATASET, NER_MODEL_PATH, HUGGINGFACE_CACHE


def train(no_cache: bool, data_config_name: str, model_path: str):
    # print(f"Loading tokenizer from {tokenizer_PATH}.")
    # tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_PATH, max_len=config.max_length)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=config.max_length)
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading and tokenizing datasets found in {NER_DATASET}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './tokcl/loader.py',
        data_config_name,
        data_dir=NER_DATASET,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=HUGGINGFACE_CACHE,
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

    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        learning_rate=5e-5,
        warmup_steps=0,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='steps',
        save_total_limit=3,
        logging_steps=10,
        eval_steps=10,
        save_steps=50,
        prediction_loss_only=False
    )

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
    trainer.save_model(model_path)

    print(f"Testing on {len(test_dataset)}.")
    pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
    print(f"{pred.metrics}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Traing script.")
    parser.add_argument("data_config_name", nargs="?", default="NER", choices=["NER", "ROLES", "BORING", "PANELIZATION"], help="Name of the dataset configuration to use.")
    parser.add_argument("--no-cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cacher.")
    args = parser.parse_args()
    no_cache = args.no_cache
    data_config_name = args.data_config_name
    model_path = Path(f"{NER_MODEL_PATH}/{data_config_name}")
    if not model_path.exists():
        model_path.mkdir()
    train(no_cache, data_config_name, str(model_path))
