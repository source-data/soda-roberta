# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy,
    BartModel, DefaultFlowCallback, EarlyStoppingCallback
)
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, GenerateMode, DatasetDict
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
import logging
from smtag.data_classes import TrainingArgumentsTOKCL
import os

logger = logging.getLogger('soda-roberta.trainer.TOKCL')

class TrainTokenClassification:
    def __init__(self,
                training_args: TrainingArgumentsTOKCL,
                loader_path: str,
                task: str,
                from_pretrained: str,
                model_type: str = 'Autoencoder',
                masked_data_collator: bool = False,
                data_dir: str = "",
                no_cache: bool = True,
                tokenizer: str = None,
                ):

        self.training_args = deepcopy(training_args)
        self.loader_path = loader_path
        self.task = task
        self.from_pretrained = from_pretrained
        self.model_type = model_type
        self.masked_data_collator = masked_data_collator
        self.data_dir = data_dir
        self.no_cache = no_cache
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.tokenizer_pretrained = tokenizer.name_or_path
        self.training_args.logging_dir = f"{RUNS_DIR}/tokcl-{self.task}-{self.from_pretrained}-{datetime.now().isoformat().replace(':','-')}"
        self.training_args.output_dir = os.path.join(training_args.output_dir,f"{self.task}_{self.from_pretrained}")

    def __call__(self):
        # Define the tokenizer
        try:
            logger.info(f"Loading the tokenizer for model {self.from_pretrained}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.from_pretrained, 
                                                            is_pretokenized=True, 
                                                            add_prefix_space=True
                                                            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_pretrained, 
                                                            is_pretokenized=True, 
                                                            add_prefix_space=True
                                                            )
            if any(x in self.tokenizer_pretrained for x in ["roberta", "gpt2"]):
                self.get_roberta = True

        # Load the dataset either from 🤗 or from local
        self.train_dataset, self.eval_dataset, self.test_dataset = self._data_loader()

        # Get the data labels
        self.id2label, self.label2id = self._get_data_labels()
        logger.info("\nTraining with {len(self.train_dataset)} examples.")
        logger.info(f"Evaluating on {len(self.eval_dataset)} examples.")
        if self.training_args.do_predict:
            logger.info(f"Testing on {len(self.test_dataset)} examples.")

        # Define the data Collator
        self.data_collator = self._get_data_collator()

        # Define the metrics to be computed
        self.compute_metrics = MetricsTOKCL(label_list=list(self.label2id.keys()))

        # Define the model 
        logger.info(f"Instantiating model for token classification {self.from_pretrained}.")
        self.model = AutoModelForTokenClassification.from_pretrained(
                                                                    self.from_pretrained,
                                                                    num_labels=len(list(self.label2id.keys())),
                                                                    max_position_embeddings=self._max_position_embeddings(),
                                                                    id2label=self.id2label,
                                                                    label2id=self.label2id,
                                                                    classifier_dropout=self.training_args.classifier_dropout,
                                                                    max_length=self.config.max_length)

        # Define the trainer
        if self.model_type == "Autoencoder":
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                data_collator=self.data_collator,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[DefaultFlowCallback,
                        EarlyStoppingCallback(early_stopping_patience=2,
                                                early_stopping_threshold=0.0)]
            )
            self.model_config = self.model.config


        elif self.model_type == "GraphRepresentation":
            # "The bare BART Model outputting raw hidden-states without any specific head on top."
            seq2seq = BartModel.from_pretrained(self.from_pretrained)  # use AutoModel instead? since LM head is provided by BecauseLM
            self.model_config = BecauseConfigForTokenClassification(
                freeze_pretrained='both',
                hidden_features=512,
                num_nodes=50,  # results into a num_nodes ** 2 latent var
                num_edge_features=6,  # not yet used
                num_node_features=10,
                sample_num_entities=20,
                sample_num_interactions=20,
                sample_num_interaction_types=3,
                sampling_iterations=100,
                alpha=1.,  # weight of adj_matrix_distro_loss
                beta=1.,  # weight of node_label_distro_loss
                gamma=0.,  # weight of the DAG loss
                seq_length=config.max_length,
                residuals=True,
                dropout=0.1,  # just to make it explicit
                classifier_dropout=0.1,
                num_labels=len(list(self.label2id.keys())),
                max_position_embeddings=config.max_length + 2  # default is 1024
            )
            self.model = BecauseTokenClassification(
                pretrained=seq2seq,
                config=self.model_config
            )
            self.trainer = MyTrainer(
                                    model=self.model,
                                    args=self.training_args,
                                    data_collator=self.data_collator,
                                    train_dataset=self.train_dataset,
                                    eval_dataset=self.eval_dataset,
                                    compute_metrics=self.compute_metrics,
                                    callbacks=[ShowExampleTOKCL(self.tokenizer)]
                                )

        else:
            raise ValueError(f"unknown model type: {self.model_type}.")

        print(f"\nTraining arguments for model type {self.model_type}:")
        print(self.model_config)
        print(self.training_args)


        # switch the Tensorboard callback to plot losses on same plot
        self.trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
        self.trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback

        logger.info(f"Training model for token classification {self.from_pretrained}.")
        self.trainer.train()

        # trainer.save_model(training_args.output_dir)

        # Define do_test
        if self.training_args.do_predict:
            logger.info(f"Testing on {len(self.test_dataset)}.")
            self.trainer.args.prediction_loss_only = False
            pred: NamedTuple = self.trainer.predict(self.test_dataset, metric_key_prefix='test')
            print(f"{pred.metrics}")

    def _data_loader(self) -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
        """
        Load the data for training, validating and testing. It will also
        send the data to the proper pipeline needed to successfully be trained.
        Returns:
            (DatasetDict, DatasetDict, DatasetDict)
        """
        logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: load_dataset('{self.loader_path}',' {self.task}')")
        data = load_dataset(self.loader_path, self.task)
        tokenized_data = data.map(
            self._tokenize_and_align_labels,
            batched=True)
        if self.masked_data_collator:
            tokenized_data.remove_columns_(['words'])
        else:
            tokenized_data.remove_columns_(['words', 'attention_mask', 'tag_mask'])
        return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']


    def _tokenize_and_align_labels(self, examples) -> DatasetDict:
        """
        Tokenizes data split into words into sub-token tokenization parts.
        Args:
            examples: batch of data from a `datasets.DatasetDict`

        Returns:
            `datasets.DatasetDict` with entries tokenized to the `AutoTokenizer`
        """
        tokenized_inputs = self.tokenizer(examples['words'],
                                          truncation=True,
                                          is_split_into_words=True,
                                          max_length=self.config.max_length)

        all_labels = examples['labels']
        new_labels = []
        tag_mask = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))
            tag_mask.append([0 if tag == 0 else 1 for tag in new_labels[-1]])

        tokenized_inputs['labels'] = new_labels
        tokenized_inputs['tag_mask'] = tag_mask

        return tokenized_inputs

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
        """
        Loads the data collator for the training. The options are the typical
        `DataCollatorForTokenClassification` or a special `DataCollationForMaskedTokenClassification`.
        Deciding between both of them can be done by setting up the parameter `--masked_data_collator`.
        Returns:
            DataCollator
        """
        if self.masked_data_collator:
            logger.info(f"""Generating the masked data collator with masking probability {self.training_args.masking_probability} 
                        and replacement prob {self.training_args.replacement_probability}""")
            self.training_args.remove_unused_columns = False
            masked_data_collator_args = self._get_masked_data_collator_args()
            data_collator = DataCollatorForMaskedTokenClassification(**masked_data_collator_args)
        else:
            logger.info("Instantiating DataCollatorForTokenClassification")
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer,
                                                               return_tensors='pt',
                                                               padding=True,
                                                               max_length=512)
        return data_collator

    def _get_data_labels(self) -> Tuple[dict, dict]:
        num_labels = self.train_dataset.info.features['labels'].feature.num_classes
        label_list = self.train_dataset.info.features['labels'].feature.names
        if self.task == "PANELIZATION":
            num_labels = 3
            label_list = ['O', 'B-PANEL_START', 'I-PANEL_START']
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        print(f"\nTraining on {num_labels} features:")
        print(", ".join(label_list))
        return id2label, label2id

    def _get_masked_data_collator_args(self) -> dict:
        """
        Generates arguments to be entered in the data collator. It works as a
        kind of default argument parser.
        Returns:
            `dict`
        """
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
              'max_length': self.config.max_length,
              'pad_to_multiple_of': None,
              'return_tensors': 'pt',
              'masking_probability': self.masking_probability,
              'replacement_probability': self.replacement_probability,
              'select_labels': self.training_args.select_labels,
        }

    def _max_position_embeddings(self) -> int:
        if any(x in self.from_pretrained for x in ["roberta", "gpt2"]) or self.get_roberta:
            return config.max_length + 2
        else:
            return config.max_length