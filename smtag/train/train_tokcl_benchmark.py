# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy,
    BartModel
)
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, GenerateMode
from ..models.experimental import (
    BecauseTokenClassification,
    BecauseConfigForTokenClassification,
)
from ..data_collator import DataCollatorForMaskedTokenClassification
from ..trainer import MyTrainer
from ..metrics import MetricsTOKCL
from ..show import ShowExampleTOKCL
from ..tb_callback import MyTensorBoardCallback
from ..config import config
from .. import LM_MODEL_PATH, TOKCL_MODEL_PATH, CACHE, RUNS_DIR
from datasets.arrow_dataset import Dataset
from typing import Dict, Union, Tuple

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
    training_args: TrainingArgumentsTOKCL,
    loader_path: str,
    task: str,
    tokenizer_name: str,
    from_pretrained: str,
    model_type: str = 'Autoencoder',
    masked_data_collator: bool = False,
    data_dir: str = "",
    no_cache: bool = True,
    do_test: bool = False,
):
    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Downloading the dataset
    train_dataset, eval_dataset, test_dataset = data_loader(loader_path, task, from_pretrained, tokenizer_name)

    id2label, label2id = get_data_labels(train_dataset)
    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")
    if do_test:
        print(f"Testing on {len(test_dataset)} examples.")

    # Defining the Data Collator
    data_collator = get_data_collator(from_pretrained,
                                      masked_data_collator,
                                      task,
                                      training_args,
                                      tokenizer)

    # Defining the metrics to be computed
    compute_metrics = MetricsTOKCL(label_list=list(label2id.keys()))

    # Defining the AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
                from_pretrained,
                num_labels=len(list(label2id.keys())),
                max_position_embeddings=max_position_embeddings(tokenizer_name),  # max_length + 2 for start/end token
                id2label=id2label,
                label2id=label2id
            )

    model_config = model.config
    print(f"\nTraining arguments for model type {model_type}:")
    print(model_config)
    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
#        callbacks=[ShowExampleTOKCL(tokenizer)]
    )

    trainer.train()


def data_loader(loader_path, task, from_pretrained, tokenizer_name):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['words'],
                                     truncation=True,
                                     is_split_into_words=True)

        all_labels = examples['labels']
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs['labels'] = new_labels
        return tokenized_inputs

    data = load_dataset(loader_path, task)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if from_pretrained in ["EMBO/bio-lm", "roberta-base"]:
        return data["train"], data['validation'], data['test']
    else:
        # Tokenize data if the data is not roberta-base tokenized
        tokenized_data = data.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=data['train'].column_names)
        return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']


def get_data_labels(dataset: Dataset) -> Tuple[dict, dict]:
    num_labels = dataset.info.features['labels'].feature.num_classes
    label_list = dataset.info.features['labels'].feature.names
    id2label, label2id = {}, {}
    for class_, label in zip(range(num_labels), label_list):
        id2label[class_] = label
        label2id[label] = class_
    print(f"\nTraining on {num_labels} features:")
    print(", ".join(label_list))
    return id2label, label2id


def max_position_embeddings(tokenizer: str) -> int:
    if tokenizer in ["roberta-base"]:
        return config.max_length + 2
    else:
        return config.max_length


def get_masked_data_collator_args(task: str, training_args: TrainingArgumentsTOKCL,
                                  tokenizer: AutoTokenizer) -> dict:
    if task == "NER":
        replacement_probability = 0.025 if training_args.replacement_probability is None else float(training_args.replacement_probability)
        # probabilistic masking
        masking_probability = 0.025 if training_args.masking_probability is None else float(training_args.masking_probability)
    elif task in ["GENEPROD_ROLES", "SMALL_MOL_ROLES"]:
        masking_probability = 1.0 if training_args.masking_probability is None else float(training_args.masking_probability)
        # pure contextual learning, all entities are masked
        replacement_probability = .0 if training_args.replacement_probability is None else float(training_args.replacement_probability)
    else:
        masking_probability = 0.0
        replacement_probability = 0.0

    return {
          'tokenizer': tokenizer,
          'padding': True,
          'max_length': 512,
          'pad_to_multiple_of': None,
          'return_tensors': 'pt',
          'masking_probability': masking_probability,
          'replacement_probability': replacement_probability,
          'select_labels': False,
    }


def shift_label(label):
    # If the label is B-XXX we change it to I-XX
    if label % 2 == 1:
        label += 1
    return label


def align_labels_with_tokens(labels, word_ids):
    """
    Expands the NER tags once the sub-word tokenization is added.
    Arguments
    ---------
    labels list[int]:
    word_ids list[int]
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            # As far as word_id matches the index of the current word
            # We append the same label
            new_labels.append(labels[word_id])
        else:
            new_labels.append(shift_label(labels[word_id]))

    return new_labels


def get_data_collator(from_pretrained, masked_data_collator, task, training_args, tokenizer):
    if from_pretrained in ["EMBO/bio-lm", "roberta-base"]:
        if masked_data_collator:
            masked_data_collator_args = get_masked_data_collator_args(task,
                                                                      training_args,
                                                                      tokenizer)
            data_collator = DataCollatorForMaskedTokenClassification(**masked_data_collator_args)
        else:
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                               return_tensors='pt')
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                           return_tensors='pt')
    return data_collator
