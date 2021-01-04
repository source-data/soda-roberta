# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
from pathlib import Path
from typing import NamedTuple
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForLanguageModeling,
)
from .trainer import MyTrainer
from .metrics import compute_metrics
from .dataset import BioDataset

from common.config import config
from common import TOKENIZER_PATH, LM_DATASET, LM_MODEL_PATH


print(f"Loading tokenizer from {TOKENIZER_PATH}.")
tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)

print(f"\nLoading and tokenizing datasets found in {LM_DATASET}.")
train_dataset = BioDataset(Path(LM_DATASET), tokenizer, subset="train")
eval_dataset = BioDataset(Path(LM_DATASET), tokenizer, subset="eval")
test_dataset = BioDataset(Path(LM_DATASET), tokenizer, subset="test")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print(f"\nTraining with {len(train_dataset)} examples.")
print(f"Evaluating on {len(eval_dataset)} examples.")

config = RobertaConfig(
    vocab_size=config.vocab_size,
    max_position_embeddings=config.max_length + 2,  # max_length + 2 for start/end token?
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)


training_args = TrainingArguments(
    output_dir=f"{LM_MODEL_PATH}",
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
    fp16=True
)

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

trainer.train()
trainer.save_model(f"{LM_MODEL_PATH}")

print(f"Testing on {len(test_dataset)}.")
pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
print(f"{pred.metrics}")
