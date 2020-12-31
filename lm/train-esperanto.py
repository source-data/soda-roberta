# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
from pathlib import Path
from typing import NamedTuple
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    TrainingArguments, DataCollatorForLanguageModeling,
    EvalPrediction
)
from .trainer import MyTrainer
from .metrics import compute_metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .dataset import BioDataset

from common.config import config
from common import TOKENIZER_PATH, DATASET, MODEL_PATH


print(f"Loading tokenizer from {TOKENIZER_PATH}.")
tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)

print(f"\nLoading and tokenizing dataset.")

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print(f"\nTraining with {len(dataset)} examples.")

config = RobertaConfig(
    vocab_size=config.vocab_size,
    max_position_embeddings=config.max_length + 2,  # max_length + 2 for start/end token?
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)


training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=False,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=False,
)

print("\nTraining arguments:")
print(training_args)

trainer = MyTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics
)

trainer.train()
