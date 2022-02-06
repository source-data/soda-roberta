# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy
)
from datasets import load_dataset, GenerateMode
from ..data_collator import DataCollatorForMaskedTokenClassification
from ..metrics import MetricsTOKCL
from ..show import ShowExampleTOCKL
from ..config import config
from .. import LM_MODEL_PATH, TOKCL_MODEL_PATH, CACHE, RUNS_DIR


# changing default values
@dataclass
class TrainingArgumentsTOKCL(TrainingArguments):
    output_dir: str = field(default=TOKCL_MODEL_PATH)
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluation_strategy: str = field(default=IntervalStrategy.STEPS)
    prediction_loss_only: bool = field(default=True)  # crucial to avoid OOM at evaluation stage!
    learning_rate: float = field(default=1e-4)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: float = field(default=10.0)
    save_total_limit: int = field(default=5)
    masking_probability: float = field(default=None)
    replacement_probability: float = field(default=None)
    select_labels: bool = field(default=False)


def train(
    no_cache: bool,
    path: str,
    data_dir: str,
    data_config_name: str,
    training_args: TrainingArgumentsTOKCL,
    tokenizer: AutoTokenizer = config.tokenizer
):

    training_args.logging_dir = f"{RUNS_DIR}/tokcl-{data_config_name}-{datetime.now().isoformat().replace(':','-')}"
    output_dir_path = Path(training_args.output_dir) / data_config_name
    if not output_dir_path.exists():
        output_dir_path.mkdir()
        print(f"Created {output_dir_path}.")
    if (data_config_name == "NER"):
        if training_args.replacement_probability is None:
            # introduce noise to scramble entities to reinforce role of context over entity identity
            training_args.replacement_probability = 0.2
        if training_args.masking_probability is None:
            training_args.masking_probability = .0
    elif data_config_name == "ROLES":
        if training_args.masking_probability is None:
            # pure contextual learning, all entities are masked
            training_args.masking_probability = 1.0
        if training_args.replacement_probability is None:
            training_args.replacement_probability = .0

    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading and tokenizing datasets found in {data_dir}.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        path,
        data_config_name,
        script_version="main",
        data_dir=data_dir,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=CACHE
    )
    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")

    if data_config_name in ["NER", "ROLES"]:
        # use our fancy data collator that randomly masks some of the inputs to enforce context learning
        training_args.remove_unused_columns = False  # we need tag_mask
        data_collator = DataCollatorForMaskedTokenClassification(
            tokenizer=tokenizer,
            max_length=config.max_length,
            masking_probability=training_args.masking_probability,
            replacement_probability=training_args.replacement_probability,
            select_labels=training_args.select_labels
        )
    else:
        # normal token classification
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            max_length=config.max_length
        )

    num_labels = train_dataset.info.features['labels'].feature.num_classes
    label_list = train_dataset.info.features['labels'].feature.names
    print(f"\nTraining on {num_labels} features:")
    print(", ".join(label_list))

    compute_metrics = MetricsTOKCL(label_list=label_list)

    model = AutoModelForTokenClassification.from_pretrained(
        LM_MODEL_PATH,
        num_labels=num_labels,
        max_position_embeddings=config.max_length + 2  # is this necessary?
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
        callbacks=[ShowExampleTOCKL(tokenizer)]
    )

    print(f"CUDA available: {torch.cuda.is_available()}")

    trainer.train()
    trainer.save_model(training_args.output_dir)

    print(f"Testing on {len(test_dataset)}.")
    trainer.args.prediction_loss_only = False
    pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
    print(f"{pred.metrics}")
