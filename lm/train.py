# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
from pathlib import Path
from typing import NamedTuple
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    EvalPrediction
)
from .trainer import MyTrainer
# from transformers.integrations import TensorBoardCallback
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .dataset import BioDataset
from datetime import datetime

from .config import config
from . import TOKENIZER_PATH, DATASET, MODEL_PATH


print(f"Loading tokenizer from {TOKENIZER_PATH}.")
tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# ) 

print(f"\nPreparing datasets found in {DATASET}.")
train_dataset = BioDataset(Path(DATASET), tokenizer, subset="train")
eval_dataset = BioDataset(Path(DATASET), tokenizer, subset="eval")
test_dataset = BioDataset(Path(DATASET), tokenizer, subset="test")

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

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    # need to ignore the padding labels -100
    set_of_labels = set(labels).remove(-100)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, labels=set_of_labels, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=False,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # eval_accumulation_steps=50,
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f"{MODEL_PATH}")

print(f"Testing on {len(test_dataset)}.")
pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
print(f"{pred.metrics}")
