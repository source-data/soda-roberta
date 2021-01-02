# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
from typing import NamedTuple
from transformers import (
    RobertaForTokenClassification, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForTokenClassification,
    EvalPrediction, Trainer
)
from datasets import load_dataset
# from datasets.utils.download_manager import GenerateMode
from common.metrics import compute_metrics
from common.config import config
from common import TOKENIZER_PATH, NER_DATASET, NER_MODEL_PATH, HUGGINGFACE_CACHE


# print(f"Loading tokenizer from {TOKENIZER_PATH}.")
# tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=config.max_length)


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
model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=num_labels)


training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=False,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # eval_accumulation_steps=50,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=False,
    label_names=['labels']
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

trainer.train()
trainer.save_model(f"{NER_MODEL_PATH}")

print(f"Testing on {len(test_dataset)}.")
pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
print(f"{pred.metrics}")
