# https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
from pathlib import Path
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
# from transformers.integrations import TensorBoardCallback
# from torch.utils.tensorboard import SummaryWriter
from .dataset import BioDataset

from . import TOKENIZER_PATH, DATASET

tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=512)

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)

train_dataset = BioDataset(Path(DATASET), tokenizer, evaluate=False)
eval_dataset = BioDataset(Path(DATASET), tokenizer, evaluate=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)


# writer = SummaryWriter()
# tb = TensorBoardCallback(writer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

trainer.train()
