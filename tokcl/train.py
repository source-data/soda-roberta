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
from common.config import config
from common import TOKENIZER_PATH, NER_DATASET, NER_MODEL_PATH, HUGGINGFACE_CACHE

from transformers import TrainerCallback, RobertaTokenizerFast
from random import randrange
import torch


class ShowExample(TrainerCallback):

    def __init__(self, tokenizer, label_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.label_list = label_list

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        N = len(eval_dataloader.dataset)
        idx = randrange(N)
        with torch.no_grad():
            inputs = eval_dataloader.dataset[idx]
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v).unsqueeze(0)
            # inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            pred = model(**inputs)
            label_idx = pred['logits'].argmax(-1)[0].cpu()
            input_ids = inputs['input_ids'][0].cpu()
        label_idx = [e.item() for e in label_idx]
        input_ids = [e.item() for e in input_ids]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        print(f"\n\nExample: {self.tokenizer.decode(input_ids)}")
        for i in range(len(input_ids)):
            print(f"{i}\t{tokens[i]}\t{self.label_list[label_idx[i]]}")


def train(no_cache: bool, data_config_name: str, model_path: str):
    # print(f"Loading tokenizer from {TOKENIZER_PATH}.")
    # tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)
    TOKENIZER = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=config.max_length)
    print(f"tokenizer vocab size: {TOKENIZER.vocab_size}")

    print(f"\nLoading and tokenizing datasets found in {NER_DATASET}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './tokcl/loader.py',
        data_config_name,
        data_dir=NER_DATASET,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=HUGGINGFACE_CACHE,
        tokenizer=TOKENIZER
    )
    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=TOKENIZER,
        padding=True,
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
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy='steps',
        save_total_limit=3,
        logging_steps=100,
        eval_steps=1,
        save_steps=100,
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
        callbacks=[ShowExample(TOKENIZER, label_list)]
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
