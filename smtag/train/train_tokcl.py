# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple, Tuple, Union, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy,
    BartModel, DefaultFlowCallback, EarlyStoppingCallback,
    AutoConfig, BertTokenizerFast
)
from transformers.integrations import TensorBoardCallback
from datasets import Dataset, load_dataset, DatasetDict
from ..models.experimental import (
    BecauseTokenClassification,
    BecauseConfigForTokenClassification,
)
from ..data_collator import DataCollatorForMaskedTokenClassification
from ..trainer import MyTrainer, ClassWeightTokenClassificationTrainer
from ..metrics import MetricsTOKCL
from ..show import ShowExampleTOKCL
from ..tb_callback import MyTensorBoardCallback
from ..config import config
from .. import LM_MODEL_PATH, TOKCL_MODEL_PATH, CACHE, RUNS_DIR
import logging
from smtag.data_classes import TrainingArgumentsTOKCL
import os
from transformers.trainer_utils import BestRun
from ray.tune.schedulers import PopulationBasedTraining, pbt
from ray import tune
from ray.tune import CLIReporter

from smtag import data_collator
from collections import Counter
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, minmax_scale
from sklearn.utils.class_weight import compute_class_weight

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
                add_prefix_space: bool = False,
                ner_labels: Union[str, List[str]] = "all"
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
        self.add_prefix_space = add_prefix_space
        self.ner_labels = ner_labels


    def __call__(self):
        # Define the tokenizer
        self.tokenizer = self._get_tokenizer()

        # Load the dataset either from 🤗 or from local
        self.train_dataset, self.eval_dataset, self.test_dataset = self._data_loader()

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
                                                                    # classifier_dropout=self.training_args.classifier_dropout,
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
                                                early_stopping_threshold=0.0),
                            ShowExampleTOKCL(self.tokenizer)]
            )

            if self.training_args.class_weights:
                weights = self._compute_class_weights()

                self.trainer = ClassWeightTokenClassificationTrainer(model=self.model,
                                                        args=self.training_args,
                                                        data_collator=self.data_collator,
                                                        train_dataset=self.train_dataset,
                                                        eval_dataset=self.eval_dataset,
                                                        compute_metrics=self.compute_metrics,
                                                        callbacks=[DefaultFlowCallback,
                                                                EarlyStoppingCallback(early_stopping_patience=2,
                                                                                        early_stopping_threshold=0.0),
                                                                    ShowExampleTOKCL(self.tokenizer)],
                                                        class_weights=weights)

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
        print(self.model.config)
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

        if self.training_args.push_to_hub:
            print(f"Uploading the model {self.trainer.model} and tokenizer {self.trainer.tokenizer} to HuggingFace")
            self.trainer.push_to_hub(commit_message="End of training")

    def _compute_class_weights(self) -> torch.tensor:
        train = self.train_dataset
        y = []
        for i in train[:]["labels"]:
            y.extend(i)

        y = np.array(y)  
        y = y[y >= 0]
        counter = Counter(y)  
        counts = []
        for key in range(len(counter.keys())):
            counts.append(counter[key])
        
        counts = np.array(counts)
        norm_weights = max(counts) / counts
        
        scaled_weights = minmax_scale(norm_weights.reshape(-1,1), feature_range=(0.2, 0.9))
        # scaled_weights = compute_class_weight("balanced", classes, y_np)    
        # print(scaled_weights)  

        return torch.tensor(scaled_weights.flatten(), 
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            dtype=torch.float,
                            )

    def _get_tokenizer(self):
        if "Megatron" in self.from_pretrained:
            tokenizer = BertTokenizerFast.from_pretrained(self.from_pretrained, 
                                                            is_pretokenized=True)
            self.get_roberta = False
        else:
            try:
                logger.info(f"Loading the tokenizer for model {self.from_pretrained}")
                tokenizer = AutoTokenizer.from_pretrained(self.from_pretrained, 
                                                                is_pretokenized=True, 
                                                                add_prefix_space=self.add_prefix_space
                                                                )
                self.get_roberta = False
            except OSError:
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_pretrained, 
                                                                is_pretokenized=True, 
                                                                add_prefix_space=self.add_prefix_space
                                                                )
                if any(x in self.tokenizer_pretrained for x in ["roberta", "gpt2"]):
                    self.get_roberta = True
        return tokenizer


    def _data_loader(self) -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
        """
        Load the data for training, validating and testing. It will also
        send the data to the proper pipeline needed to successfully be trained.
        Returns:
            (DatasetDict, DatasetDict, DatasetDict)
        """
        logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: load_dataset('{self.loader_path}',' {self.task}')")
        data = load_dataset(self.loader_path, self.task, ignore_verifications=True)

        # Get the data labels
        self.dataset_id2label, self.dataset_label2id = self._get_data_labels(data["train"])
        self.id2label, self.label2id = self._generate_new_label_dict()
        logger.info(f"Training with {len(data['train'])} examples.")
        logger.info(f"Evaluating on {len(data['validation'])} examples.")

        if self.loader_path == "EMBO/sd-nlp":
            tokenized_data = deepcopy(data)
            if not self.masked_data_collator:
                tokenized_data.remove_columns_(['tag_mask'])
        else:
            if self.masked_data_collator:
                columns_to_remove = ['words']
            else:
                columns_to_remove = ['words', 'tag_mask']
            tokenized_data = data.map(
                self._tokenize_and_align_labels,
                batched=True,
                remove_columns=columns_to_remove)

        if (self.ner_labels != "all") or (self.ner_labels not in ["all"]):
            tokenized_data = tokenized_data.map(
                self._substitute_training_labels,
                batched=True)

        return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']

    def _substitute_training_labels(self, examples):
        
        all_labels = examples['labels']
        new_labels = []
        new_tag_mask = []
        for labels in all_labels:
            new_labels_sentence = []
            for label in labels:
                if label == -100:
                    new_labels_sentence.append(label)
                elif self.dataset_id2label[label] in list(self.id2label.values()):
                    new_labels_sentence.append(self.label2id[self.dataset_id2label[label]])
                else:
                    new_labels_sentence.append(0)
            new_labels.append(new_labels_sentence)
            new_tag_mask.append([0 if tag == 0 else 1 for tag in new_labels[-1]])

        examples['labels'] = new_labels
        examples['tag_mask'] = new_tag_mask

        return examples
        
    def _generate_new_label_dict(self):
        id2label, label2id = {}, {}
        if (self.ner_labels == ["all"]) or (self.ner_labels == "all"):
            id2label, label2id = self.dataset_id2label, self.dataset_label2id
        else:
            new_labels = ["O"]
            for label in self.ner_labels:
                new_labels.append(f"B-{label}")
                new_labels.append(f"I-{label}")
            for i, label in enumerate(new_labels):
                id2label[i] = label
                label2id[label] = i
        return id2label, label2id
        

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

    def _get_data_labels(self, data: Dataset) -> Tuple[dict, dict]:
        num_labels = data.info.features['labels'].feature.num_classes
        label_list = data.info.features['labels'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        print(f"The data set has {num_labels} features: {label_list}")
        print(f"\nTraining on {len(self.ner_labels)} features: {self.ner_labels}")
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
            self.replacement_probability = 0.0 if self.training_args.replacement_probability is None else float(self.training_args.replacement_probability)
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


class HpSearchForTokenClassification(TrainTokenClassification):
    def __init__(self,
                smoke_test: bool = False,
                gpus_per_trial: int = 0,
                cpus_per_trial: int = 0,
                hp_tune_samples: int = 8,
                hp_search_config: dict = {},
                hp_search_scheduler: pbt.PopulationBasedTraining = PopulationBasedTraining(),
                hp_search_reporter: tune.progress_reporter.CLIReporter = CLIReporter(),
                hp_experiment_name: str = "tune_transformer_pbt",
                hp_local_dir: str = "/app/ray_results/",
                **kw
            ):
        self.smoke_test = smoke_test
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.hp_tune_samples = hp_tune_samples
        self.hp_search_config = hp_search_config
        self.hp_search_scheduler = hp_search_scheduler
        self.hp_search_reporter = hp_search_reporter
        self.hp_experiment_name = hp_experiment_name
        self.hp_local_dir = hp_local_dir
        super(HpSearchForTokenClassification, self).__init__(**kw)
        self.training_args.logging_dir = f"{RUNS_DIR}/tokcl-{self.task}-{self.from_pretrained}-{datetime.now().isoformat().replace(':','-')}"
        self.training_args.output_dir = os.path.join(
                                                self.training_args.output_dir,
                                                f"{self.task}_{self.from_pretrained}"
                                                )
        self.model_name = self.from_pretrained if not self.smoke_test else "sshleifer/tiny-distilroberta-base"

    def _get_model(self):
        """Model initializer function. This is just a call to be used by the
        automated hyperparameter search.

        Returns:
            AutoModelForTokenClassification: Any of the AutoModelForTokenClassification available in 🤗.
        """
                # Checking the hyperparameter tuning
        config = AutoConfig.from_pretrained(
            self.model_name, 
            num_labels=len(list(self.label2id.keys())), 
        )        
        return AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            config=config
            )

    def _run_hyperparam_search(self) -> BestRun:
        """Runs the hyperparameter search to find the best hyper parameters
        for the model. It uses the Ray backend.

        Returns:
            trainer_utils.BestRun: All the information about the best model run.
        """
        # Define the tokenizer
        self.tokenizer = self._get_tokenizer()

        # Load the dataset either from 🤗 or from local
        self.train_dataset, self.eval_dataset, _ = self._data_loader()

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

        training_args = TrainingArguments(
            output_dir=os.path.join(self.training_args.output_dir,f"hp_tuning_{self.task}_{self.from_pretrained}"),
            learning_rate=1e-5,  # config
            do_train=True,
            do_eval=True,
            no_cuda=self.gpus_per_trial <= 0,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            num_train_epochs=2,  # config
            max_steps=-1,
            per_device_train_batch_size=16,  # config
            per_device_eval_batch_size=16,  # config
            warmup_steps=0,
            weight_decay=0.1,  # config
            logging_dir="./logs",
            skip_memory_metrics=True,
            report_to="none",
            disable_tqdm=self.training_args.disable_tqdm
        )
        trainer = Trainer(
            model_init=self._get_model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        logger.info(f"Hyperparameter search will take place.")
        best_model =  trainer.hyperparameter_search(
                                hp_space=lambda _: self.hp_search_config,
                                backend="ray",
                                n_trials=self.hp_tune_samples,
                                resources_per_trial={"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial},
                                scheduler=self.hp_search_scheduler,
                                keep_checkpoints_num=1,
                                checkpoint_score_attr="training_iteration",
                                stop={"training_iteration": 1} if self.smoke_test else None,
                                progress_reporter=self.hp_search_reporter,
                                local_dir=self.hp_local_dir,
                                name=self.hp_experiment_name,
                                log_to_file=True,
                            )

        return best_model
