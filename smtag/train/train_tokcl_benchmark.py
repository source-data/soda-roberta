# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import torch
import pickle
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy,
    RobertaConfig, BertConfig
)
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, GenerateMode
from ..data_collator import DataCollatorForMaskedTokenClassification
from ..trainer import MyTrainer
from ..metrics import MetricsTOKCL
from ..show import ShowExampleTOKCL
from ..tb_callback import MyTensorBoardCallback
from ..config import config
from .. import LM_MODEL_PATH, TOKCL_MODEL_PATH, CACHE, RUNS_DIR
from datasets.arrow_dataset import Dataset
from typing import Dict, Union, Tuple
from torch.utils.data import DataLoader
from datasets import load_metric
from os.path import exists
import json
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


class TrainModel:
    def __init__(self, training_args: TrainingArgumentsTOKCL,
                 loader_path: str,
                 task: str,
                 tokenizer_name: str,
                 from_pretrained: str,
                 model_type: str = 'Autoencoder',
                 masked_data_collator: bool = False,
                 data_dir: str = "",
                 no_cache: bool = True,
                 do_test: bool = False,
                 dropout: float = 0.2,
                 hidden_size_multiple: int = 50,
                 file_: str = "./benchmarking_results.json"
                 ):

        self.training_args = training_args
        self.loader_path = loader_path
        self.task = task
        self.tokenizer_name = tokenizer_name
        self.from_pretrained = from_pretrained
        self.model_type = model_type
        self.masked_data_collator = masked_data_collator
        self.data_dir = data_dir
        self.no_cache = no_cache
        self.do_test = do_test
        self.dropout = dropout
        self.hidden_size = hidden_size_multiple
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.file_ = file_

        # Define the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def __call__(self):

        # Downloading the dataset
        self.train_dataset, self.eval_dataset, self.test_dataset = self._data_loader()

        self.id2label, self.label2id = self._get_data_labels()
        print(f"\nTraining with {len(self.train_dataset)} examples.")
        print(f"Evaluating on {len(self.eval_dataset)} examples.")
        if self.do_test:
            print(f"Testing on {len(self.test_dataset)} examples.")

        # Defining the Data Collator
        self.data_collator = self._get_data_collator()

        # Defining the metrics to be computed
        self.compute_metrics = MetricsTOKCL(label_list=list(self.label2id.keys()))

        # Defining the AutoModelForTokenClassification
        # Here I should check if using model config would be better
        # Note that in that case I would be able to control part
        # of the training hyperparameters
        if self.from_pretrained in ['roberta-base', 'EMBO/bio-lm']:
            self.hidden_size = self.hidden_size * RobertaConfig().num_attention_heads
        elif self.from_pretrained in ['bert-base-cased', 'bert-base-uncased',
                                      'dmis-lab/biobert-base-cased-v1.2',
                                      'dmis-lab/biobert-base-cased-v1.1',
                                      'dmis-lab/biobert-large-cased-v1.1',
                                      'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                                      'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext']:
            self.hidden_size = self.hidden_size * BertConfig().num_attention_heads

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.from_pretrained,
            num_labels=len(list(self.label2id.keys())),
            max_position_embeddings=self._max_position_embeddings(),
            id2label=self.id2label,
            label2id=self.label2id,
            classifier_dropout=self.dropout,
            hidden_size=self.hidden_size,
        )

        model_config = self.model.config
        print(f"\nTraining arguments for model type {self.model_type}:")
        print(model_config)
        print(self.training_args)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            # callbacks=[TensorBoardCallback]
        )
        # switch the Tensorboard callback to plot losses on same plot
        self.trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
        self.trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback

        print(f"CUDA available: {torch.cuda.is_available()}")

        self.trainer.train()

        if self.do_test:
            self._run_test()

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['words'],
                                          truncation=True,
                                          is_split_into_words=True)

        all_labels = examples['labels']
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        tokenized_inputs['labels'] = new_labels
        return tokenized_inputs

    def _data_loader(self):
        data = load_dataset(self.loader_path, self.task)
        if self.from_pretrained in ["EMBO/bio-lm", "roberta-base"]:
            return data["train"], data['validation'], data['test']
        else:
            # Tokenize data if the data is not roberta-base tokenized
            tokenized_data = data.map(
                self._tokenize_and_align_labels,
                batched=True,
                remove_columns=data['train'].column_names)
            return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']

    def _get_data_labels(self) -> Tuple[dict, dict]:
        num_labels = self.train_dataset.info.features['labels'].feature.num_classes
        label_list = self.train_dataset.info.features['labels'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        print(f"\nTraining on {num_labels} features:")
        print(", ".join(label_list))
        return id2label, label2id

    def _max_position_embeddings(self) -> int:
        if self.tokenizer in ["roberta-base"]:
            return config.max_length + 2
        else:
            return config.max_length

    def _get_masked_data_collator_args(self) -> dict:
        if self.task == "NER":
            self.replacement_probability = 0.025 if self.training_args.replacement_probability is None else float(self.training_args.replacement_probability)
            # probabilistic masking
            self.masking_probability = 0.025 if self.training_args.masking_probability is None else float(self.training_args.masking_probability)
        elif self.task in ["GENEPROD_ROLES", "SMALL_MOL_ROLES"]:
            self.masking_probability = 1.0 if self.training_args.masking_probability is None else float(self.training_args.masking_probability)
            # pure contextual learning, all entities are masked
            self.replacement_probability = .0 if self.training_args.replacement_probability is None else float(self.training_args.replacement_probability)
        else:
            self.masking_probability = 0.0
            self.replacement_probability = 0.0

        return {
              'tokenizer': self.tokenizer,
              'padding': True,
              'max_length': 512,
              'pad_to_multiple_of': None,
              'return_tensors': 'pt',
              'masking_probability': self.masking_probability,
              'replacement_probability': self.replacement_probability,
              'select_labels': False,
        }

    @staticmethod
    def _shift_label(label):
        # If the label is B-XXX we change it to I-XX
        if label % 2 == 1:
            label += 1
        return label

    def _align_labels_with_tokens(self, labels, word_ids):
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
                new_labels.append(self._shift_label(labels[word_id]))

        return new_labels

    def _get_data_collator(self):
        # if self.from_pretrained in ["EMBO/bio-lm", "roberta-base"]:
        if self.masked_data_collator:
            masked_data_collator_args = self._get_masked_data_collator_args()
            data_collator = DataCollatorForMaskedTokenClassification(**masked_data_collator_args)
        else:
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer,
                                                               return_tensors='pt')
        # else:
        #     data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer,
        #                                                        return_tensors='pt')
        return data_collator

    def _run_test(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=64, collate_fn=self.data_collator)
        metric = load_metric('seqeval')
        self.model.eval()

        for batch in test_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            batch_true_labels, batch_predictions = [], []
            for i, sentence in enumerate(batch['labels']):
                true_label_list, predictions_list = [], []
                for true_label, prediction in zip(sentence.tolist(), predictions[i].tolist()):
                    if true_label != -100:
                        true_label_list.append(self.model.config.id2label[true_label])
                        predictions_list.append(self.model.config.id2label[prediction])

                batch_true_labels.append(true_label_list)
                batch_predictions.append(predictions_list)

            metric.add_batch(predictions=batch_predictions,
                             references=batch_true_labels)

        self.test_results = metric.compute()
        print(100*"-")
        print(f"Test results.")
        print(100*"-")
        print(self.test_results)

    def save_benchmark_results(self):

        output_data = {
            "date": datetime.today(),
            "model_name": self.training_args.hub_model_id,
            "pretrained_model": self.from_pretrained,
            "base_model" : self.model.base_model_prefix,
            "hidden_size" : self.model.classifier.in_features,
            "attention_heads": self.model.config.num_attention_heads,
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "base_model_parameters": self.model.base_model.num_parameters(),
            "masked_data_collator": self.masked_data_collator,
            "dropout": self.dropout,
            "vocab_size": self.tokenizer.vocab_size,
            "task": self.task,
            "id2label": self.id2label,
            "training_epochs": self.training_args.num_train_epochs,
            "training_examples": len(self.train_dataset),
            "training_steps": len(self.train_dataset) * self.training_args.num_train_epochs,
            "learning_rate_init": self.training_args.learning_rate,
            "learning_rate_scheduled": self.training_args.lr_scheduler_type,
            "training_batch_size": self.training_args.per_device_train_batch_size,
            "accuracy_metrics": self.test_results
        }

        if exists(self.file_):
            with open(self.file_, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                data["test_results"].append(output_data)
        else:
            data = {'test_results': [output_data]}

        with open(self.file_, 'wb') as pkl_file:
            pickle.dump(data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)





