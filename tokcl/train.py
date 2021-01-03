# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
import torch
from transformers import (
    RobertaForTokenClassification, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer
)
from datasets import load_dataset
# from datasets.utils.download_manager import GenerateMode
from .metrics import MetricsComputer
from common.config import config
from common import TOKENIZER_PATH, NER_DATASET, NER_MODEL_PATH, HUGGINGFACE_CACHE


# print(f"Loading tokenizer from {TOKENIZER_PATH}.")
# tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=config.max_length)
print(f"tokenizer vocab size: {tokenizer.vocab_size}")


print(f"\nLoading and tokenizing datasets found in {NER_DATASET}.")
train_dataset, eval_dataset, test_dataset = load_dataset(
    './tokcl/dataset.py',
    'NER',
    split=["train", "validation", "test"],
    # download_mode=GenerateMode.FORCE_REDOWNLOAD,
    cache_dir=HUGGINGFACE_CACHE
)
print(f"\nTraining with {len(train_dataset)} examples.")
print(f"Evaluating on {len(eval_dataset)} examples.")


data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    max_length=config.max_length
)

num_labels = train_dataset.info.features['labels'].feature.num_classes
label_list = train_dataset.info.features['labels'].feature.names
compute_metrics = MetricsComputer(label_list=label_list)

model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=False,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy='steps',
    eval_steps=1,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=False,
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
)

print(f"CUDA available: {torch.cuda.is_available()}")

trainer.train()
trainer.save_model(f"{NER_MODEL_PATH}")

print(f"Testing on {len(test_dataset)}.")
pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
print(f"{pred.metrics}")
