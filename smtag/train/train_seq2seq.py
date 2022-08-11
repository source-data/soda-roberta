# Everything that goes here assumes a dataset that is ready to go. Or I can fix I wanrt the DS as input###output and then I format it on the way I want for GPT or HF
import sre_compile
from datasets import DatasetDict, Dataset, load_dataset
import logging
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from smtag.data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq, Gpt3ModelParam
from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleTextGeneration
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer
    )
from transformers.integrations import TensorBoardCallback
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import itertools 

logger = logging.getLogger('soda-roberta.train_seq2seq.HfSeq2SeqTrainer')

class HfSeq2SeqTrainer:
    def __init__(self, 
                 # DATA AND MODELS
                 datapath: str,
                 task: str,
                 task_type: str = "copy_tag",
                 labels_list = "all",
                 delimiter: str = "###tt9HHSlkWoUM###",
                 base_model: str = "t5-base",
                 from_local_checkpoint: str = None,
                 # SPECIAL FOR NER
                 prompt_init: str = "Do NER on the entities",
                 prompt_end: str = "\n\nEND_INPUT\n\n",
                 generate_end: str = "[END]",
                 # DATA GENERATION
                 split: List[float] = [0.8, 0.1, 0.1],
                 skip_lines: int = 0,
                 # TOKENIZER PARAMETERS
                 max_input_length: int = 512,
                 max_target_length: int = 512,
                 # MODEL PARAMETERS
                 model_param: ModelConfigSeq2Seq = ModelConfigSeq2Seq(),
                 # TRAINING PARAMETERS
                 training_args: TrainingArgumentsSeq2Seq = TrainingArgumentsSeq2Seq()
    ):
        """The class will accept `*.csv` files encoded as string lines of
        input separator target, and model names (local or in the 🤗 Hub) to generate
        training. This training is intended for Seq2Seq models. Currently, only the 
        `ConditionalGeneration` task is accepted. I the future other tasks, included
        in the Seq2Seq models of 🤗 will be added.

        Args:
            datapath (str): local path or 🤗 Hub path to the data.
            delimiter (str, optional): delimiter separating input and targets in the dataset. Defaults to "###tt9HHSlkWoUM###".
            base_model (str, optional): base 🤗 model. Defaults to "t5-base".
            from_local_checkpoint (str, optional): If a local stored model is to be used. Defaults to None.
            split (List[float], optional): Percentages of data splits for train, eval, test. Defaults to [0.8, 0.1, 0.1].
            skip_lines (int, optional): Lines at the top of the file to be skipped. Defaults to 0.
            max_input_length (int, optional): Maximum length accepted for the input sequences. Defaults to 512.
            max_target_length (int, optional): Maximum length to be generated by the `generate` method of the models. Defaults to 512.
            model_param (ModelConfigSeq2Seq, optional): 🤗 parameters for Seq2Seq models. Defaults to ModelConfigSeq2Seq().
            training_args (TrainingArgumentsSeq2Seq, optional): 🤗 training arguments for Seq2Seq tasks. Defaults to TrainingArgumentsSeq2Seq().
        """
        self.datapath = datapath
        self.task = task
        self.task_type = task_type
        self.labels_list = labels_list
        self.delimiter = delimiter
        self.split = split
        self.skip_lines = skip_lines
        self.base_model = base_model
        self.from_local_checkpoint = from_local_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_config = model_param
        self.training_args = training_args
        self.prompt_init =prompt_init
        self.prompt_end = prompt_end
        self.generate_end = generate_end

    def __call__(self):
        if self.task != "NER":
            try:
                logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: {self.datapath}")
                self.dataset = load_dataset(self.datapath)
            except FileNotFoundError:
                assert self.datapath.split('.')[-1] in ['csv', 'txt', 'tsv'], \
                    logger.warning("""The data format is not supported. Please upload a file with format {'csv', 'txt', 'tsv'}
                            or write a valid path to a dataset in HuggingFace 🤗 Hub.""")
                logger.info(f"Obtaining data from the local file: {self.datapath}")
                self.dataset = self._load_dataset_into_hf()
        else:
            convert = FromIob2seq2seq(
                data_loc=self.datapath, 
                labels=self.labels_list,
                prompt_init=self.prompt_init,
                prompt_end =self.prompt_end,
                generate_end=self.generate_end,
                task_type=self.task_type,
                                    )
            self.dataset = convert()
            
        self.tokenizer, self.model = self._get_model_and_tokenizer()

        logger.info(f"""Tokenizing the data using the tokenizer of model {self.base_model}\n
                        {self.tokenizer}""")

        self.tokenized_dataset = self._tokenize_data()

        logger.info(f"""Preparing the data collator""")
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model=self.model,
                                               padding=True,
                                               return_tensors='pt')

        
        try:
            eval_ds = self.tokenized_dataset['eval']
        except KeyError:
            eval_ds = self.tokenized_dataset['validation']

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            compute_metrics=None,
            callbacks=[ShowExampleTextGeneration(self.tokenizer, **{
                "max_sentence_length": self.training_args.generation_max_length,
                "model_config": self.model_config.__dict__
                    }
                )
            ]
        )

        if self.training_args.do_train:
            trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
            trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback
            trainer.train()

        if self.training_args.do_predict:    
            test_data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               padding=True,
                                               return_tensors='pt')
            columns_to_remove = set(self.tokenized_dataset['test'].column_names) - set(["input_ids", "attention_mask"])
            self.test_ds = self.tokenized_dataset['test'].remove_columns(columns_to_remove)
            logger.info("Preparing the model evaluation in the testing dataset.")
            test_dataloader = DataLoader(
                self.test_ds, 
                collate_fn=test_data_collator, 
                batch_size=self.training_args.per_device_eval_batch_size
                )
            self.model.eval()
            self.output_text = []
            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    batch["attention_mask"] = batch["attention_mask"].to(device='cuda')
                    batch["input_ids"] = batch["input_ids"].to(device='cuda')
                    output_tokens = self.model.generate(**batch)
                    output_string = self.tokenizer.batch_decode(output_tokens)
                    for example in output_string:
                        self.output_text.append(example)

        self.output_ds = self.tokenized_dataset["test"]
        self.output_ds = self.output_ds.add_column("generated_text", self.output_text)
        self.output_ds.to_csv("./data/seq2seq/predictions_NER_seq2seq.csv")
        # self.tokenized_dataset["test"].to_csv("./data/seq2seq/predictions_NER_seq2seq.csv")
        return self.dataset, self.tokenizer, self.model, self.tokenized_dataset

    def _tokenize_data(self):
        return self.dataset.map(self._preprocess_data,
                                batched=True)

    def _get_model_and_tokenizer(self):
        if self.from_local_checkpoint:
            logger.info(f"Downloading the model based on: {self.base_model} and checkpoint {self.from_local_checkpoint}")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.from_local_checkpoint, **self.model_config.__dict__)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        else:
            logger.info(f"Downloading the model based on: {self.base_model}")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model, **self.model_config.__dict__)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer, model

    def _load_dataset_into_hf(self) -> DatasetDict:
        """
        Loads a file encoded as a `csv` into a format that can be used
        by the `Seq2Seq` class.
        The `Seq2Seq` class accepts HuggingFace 🤗 datasets with features
        called `input` and `target`. Both features encoded as `str`.

        :param datapath: `str` path to the `csv` file
        :param delimiter: `str` delimiter between the two data fields
        :param split: `list` of `float` percentage of data splits for training, validation, and test
        :param skip_lines: `int` number of lines to skip at the beginning of the file

        :return: datasets.DatasetDict with the data split into `train`, `test`, and `eval` sets
        """

        assert sum(self.split) == 1, logging.critical(f"""The numbers of the split argument must sum up to one. They are 
                                    {self.split}, summing up to {sum(self.split)}""")
        assert len(self.split) == 3, logging.critical(f"""The length of the split argument must be 3. You have 
                                    {len(self.split)} elements in the list.""")
    
        result = {
                    'train': {'input': [],'target': [], 'fig_id':[], 'doi': []},
                    'test': {'input': [],'target': [], 'fig_id':[], 'doi': []},
                    'eval': {'input': [],'target': [], 'fig_id':[], 'doi': []}
                    }
        logger.info(f"Reading lines from file {self.datapath}")
        dataframe = pd.read_csv(self.datapath)
        dataframe = dataframe.dropna()
        for _, row in dataframe.iterrows():
            split = row["subset"]
            data_input = row["input_text"]
            data_output = row["output_text"]
            doi = row["doi"]
            f_id = row["f_id"]

            result[split]["input"].append(data_input)
            result[split]["target"].append(data_output)
            result[split]["fig_id"].append(doi)
            result[split]["doi"].append(f_id)

        return DatasetDict(
                            {
                                'train': Dataset.from_dict(result['train']),
                                'eval': Dataset.from_dict(result['eval']),
                                'test': Dataset.from_dict(result['test'])
                            }
                        )

    def _preprocess_data(self, examples):
        """
        Method that will be used to tokenize the input and target data on a way
        that can be used by the Seq2Seq model for training and inference.
        Method to be used with `Dataset.map` or `DatasetDict.map`.
        :param examples: iterable elements of the dataset
        :return: tokenized examples for a `Dataset.map` or `DatasetDict.map`.
        """
        # input_ = list(map(lambda orig_string: self.task + orig_string, examples['input']))
        model_inputs = self.tokenizer(
            examples['input'], max_length=self.max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target"], max_length=self.max_target_length, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs


class Gpt3FineTuner(HfSeq2SeqTrainer):
    def __init__(self, datapath, end_prompt_token = "\n\n###\n\n"):
        """Converts a `*.csv` file of the formar input separator target into
        a format accepted by the OpenAI GPT API.

        Args:
            datapath (_type_): path to the `*.csv` file
            end_prompt_token (str, optional): Token to be added at the end of each input
            to indicate that the model must now begin the prediction. 
            Defaults to "\n\n###\n\n".
        """
        super().__init__(datapath)
        self.end_prompt_token = end_prompt_token
        self.folder, self.file_name = os.path.split(self.datapath)

    def __call__(self):
        self.dataset = self._load_dataset_into_hf()

        self.dataset = self.dataset.map(self._convert_to_gpt, batched=False)
        self.dataset.remove_columns_(["input", "target"])

        self.dataset["train"].to_json(os.path.join(self.folder, "gpt_ready_train.jsonl"), lines=True, orient="records")
        self.dataset["eval"].to_json(os.path.join(self.folder, "gpt_ready_eval.jsonl"), lines=True, orient="records")
        self.dataset["test"].to_json(os.path.join(self.folder, "gpt_ready_test.jsonl"), lines=True, orient="records")

        return self.dataset

    def _convert_to_gpt(self, example):
        result = {
            "prompt": f'''{example["input"]} {self.end_prompt_token} ''',
            "completion": " " + example["target"]
        }
        return result


class FromIob2seq2seq:

    def __init__(self, 
                data_loc: str = "EMBO/sd-nlp-non-tokenized", 
                labels: List[str] = ["all"],
                prompt_init: str = "Do NER on the entities",
                prompt_end: str = "\n\nEND_INPUT\n\n",
                generate_end: str = "[END]",
                task_type: str = "list",
                ):
        self.task_type = task_type
        self.loader_path = data_loc
        self.label_list = self._get_labels_list(labels)
        self.prompt_init = f"{prompt_init} "
        self.prompt_end = prompt_end
        self.generate_end = generate_end
        self.data = load_dataset(self.loader_path, "NER", ignore_verifications=True)
        self.id2label, self.label2id = self._get_data_labels()

    def __call__(self):
        seq2seq_data = self.data.map(
                                self._iob2text,
                                batched=True
                                )
        return seq2seq_data

    def _iob2text(self, examples) -> dict:
        input_text = []
        output_text = []

        for words, labels in zip(examples["words"], examples["labels"]):
            input_text.append(f"{self.prompt_init} {' '.join(words)} {self.prompt_end}")
            if self.task_type == "list":
                output_text.append(f"{self._get_output_text(words, labels)} {self.generate_end}")
            if self.task_type == "copy_tag":
                output_text.append(f"{self._copy_and_tag(words, labels)} {self.generate_end}")    
        examples["input"] = input_text
        examples["target"] = output_text
        
        return {"input": examples["input"], "labels": examples["target"]}

    def _get_output_text(self, words: list, labels: list) -> str:
        output_text = ""
        for entity_type in self.label_list:
            entities = []
            in_entity = False
            for w, l in zip(words, labels):
                if (self.id2label[l] == f"B-{entity_type}") and (in_entity is False):
                    entity_text = w
                    in_entity = True
                elif (self.id2label[l] == f"B-{entity_type}") and (in_entity is True):
                    entities.append(entity_text)
                    entity_text = w
                    in_entity = True
                elif (self.id2label[l] == f"I-{entity_type}") and (in_entity is True):
                    entity_text += w
                elif (self.id2label[l] == "O") and (in_entity is True):
                    entities.append(entity_text)
                    in_entity = False
                elif (self.id2label[l] == "O") and (in_entity is False):
                    continue
                else:
                    continue

            output_text += f"{entity_type}: {', '.join(list(dict.fromkeys(entities)))} \n"
        return f"{output_text} {self.generate_end}"

    def _get_data_labels(self) -> Tuple[dict, dict]:
        num_labels = self.data["train"].info.features['labels'].feature.num_classes
        label_list = self.data["train"].info.features['labels'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        return id2label, label2id

    @staticmethod
    def _get_labels_list(list_):
        if list_ == ["all"]:
            return ["GENEPROD", "TISSUE", "ORGANISM", "SMALL_MOLECULE", "EXP_ASSAY", "CELL", "SUBCELLULAR"]
        else:
            return list_

    def _copy_and_tag(self, words: list, labels: list) -> str:
        output_string = ""
        inside_label = None

        for idx, (word, label) in enumerate(zip(words, labels)):
            if (label == 0) and not inside_label:
                output_string += f"{word} "
            elif (label == 0) and inside_label:
                output_string += f"</{inside_label}> {word} "
                inside_label = None
            elif self.id2label[label].startswith("B-") and not inside_label:
                if self.id2label[label].split('-')[1] in self.label_list:
                    output_string += f"<{self.id2label[label].split('-')[1]}> {word} "
                    inside_label = self.id2label[label].split('-')[1]
                if self.id2label[label].split('-')[1] not in self.label_list:
                    output_string += f"{word} "
            elif self.id2label[label].startswith("B-") and inside_label:
                if self.id2label[label].split('-')[1] in self.label_list:
                    output_string += f"</{inside_label}> <{self.id2label[label].split('-')[1]}> {word} "
                    inside_label = self.id2label[label].split('-')[1]
                if self.id2label[label].split('-')[1] not in self.label_list:
                    output_string += f"</{inside_label}> {word} "
                    inside_label = None
            elif self.id2label[label].startswith("I-"):
                output_string += f"{word } "
            else:
                raise NotImplementedError
        return output_string