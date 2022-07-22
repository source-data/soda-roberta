# Everything that goes here assumes a dataset that is ready to go. Or I can fix I wanrt the DS as input###output and then I format it on the way I want for GPT or HF
from datasets import DatasetDict, Dataset, load_dataset
import logging
import numpy as np
from typing import List
from smtag.data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq, Gpt3ModelParam
from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleTextGeneration
from transformers import (
    AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainer
    )
from transformers.integrations import TensorBoardCallback
import os

logger = logging.getLogger('soda-roberta.train_seq2seq.HfSeq2SeqTrainer')

class HfSeq2SeqTrainer:
    def __init__(self, 
                 # DATA AND MODELS
                 datapath: str,
                 delimiter: str = "###tt9HHSlkWoUM###",
                 base_model: str = "t5-base",
                 from_local_checkpoint: str = None,
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
        self.datapath = datapath
        self.delimiter = delimiter
        self.split = split
        self.skip_lines = skip_lines
        self.base_model = base_model
        self.from_local_checkpoint = from_local_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_config = model_param
        self.training_args = training_args

    def __call__(self):
        try:
            logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: {self.datapath}")
            self.dataset = load_dataset(self.datapath)
        except FileNotFoundError:
            assert self.datapath.split('.')[-1] in ['csv', 'txt', 'tsv'], \
                logger.warning("""The data format is not supported. Please upload a file with format {'csv', 'txt', 'tsv'}
                        or write a valid path to a dataset in HuggingFace 🤗 Hub.""")
            logger.info(f"Obtaining data from the local file: {self.datapath}")
            self.dataset = self._load_dataset_into_hf()
            
        self.tokenizer, self.model = self._get_model_and_tokenizer()

        logger.info(f"""Tokenizing the data using the tokenizer of model {self.base_model}\n
                        {self.tokenizer}""")
        self.tokenized_dataset = self._tokenize_data()

        logger.info(f"""Preparing the data collator""")
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model=self.model,
                                               padding=True,
                                               return_tensors='pt')
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['eval'],
            tokenizer=self.tokenizer,
            callbacks=[ShowExampleTextGeneration(self.tokenizer, **{"max_sentence_length": self.training_args.generation_max_length})]
        )

        if self.training_args.do_train:
            trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
            trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback
            trainer.train()
            print(self.training_args)

        return self.dataset, self.tokenizer, self.model, self.tokenized_dataset

    def _tokenize_data(self):
        return self.dataset.map(self._preprocess_data,
                                batched=True)

    def _get_model_and_tokenizer(self):
        if self.from_local_checkpoint:
            logger.info(f"Downloading the model based on: {self.base_model} and checkpoint {self.from_local_checkpoint}")
            if 'bart' in self.base_model:
                model = BartForConditionalGeneration.from_pretrained(self.from_local_checkpoint,
                                                                          **self.model_config.__dict__)
            elif 't5' in self.base_model:
                model = T5ForConditionalGeneration.from_pretrained(self.from_local_checkpoint,
                                                                        **self.model_config.__dict__)
            else:
                raise ValueError(f"""Please select a model that is compatible with the 
                                    conditional generation task: {['bart', 't5']}.""")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        else:
            logger.info(f"Downloading the model based on: {self.base_model}")
            if 'bart' in self.base_model:
                model = BartForConditionalGeneration.from_pretrained(self.base_model,
                                                                          **self.model_config.__dict__)
            elif 't5' in self.base_model:
                model = T5ForConditionalGeneration.from_pretrained(self.base_model,
                                                                        **self.model_config.__dict__)
            else:
                raise ValueError(f"""Please select a model that is compatible wit the 
                                    conditional generation task: {['bart', 't5']}.""")
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
    
        data_output = {'train': {'input': [],'target': []},
                    'test': {'input': [], 'target': []},
                    'eval': {'input': [], 'target': []}}
        logger.info(f"Reading lines from file {self.datapath}")
        with open(self.datapath, 'r') as file_:
            for line in file_.readlines()[self.skip_lines:]:
                data_pair = line.strip().split(self.delimiter)
                choice = np.random.choice(["train", "eval", "test"], p=self.split)
                try:
                    input_, target = data_pair
                    data_output[choice]['input'].append(input_.strip().replace('\ufeff', '').replace('\xa0', ' '))
                    data_output[choice]['target'].append(target.strip().replace('\ufeff', '').replace('\xa0', ' '))
                except ValueError:
                    logger.warning(f"""Wrong number of task--input in line. Possible lack of 
                                    delimiter. Skipping line to next one. {line}""")
                    continue

        return DatasetDict(
                            {
                                'train': Dataset.from_dict(data_output['train']),
                                'eval': Dataset.from_dict(data_output['eval']),
                                'test': Dataset.from_dict(data_output['test'])
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

# This will include HF models and the GPT-3 API call

# I will do 2 classes. One for each with their own data classes

# I must provide a function to go from a plain text file of input###output to gpt and HF format

class Gpt3FineTuner(HfSeq2SeqTrainer):
    def __init__(self, datapath, end_prompt_token = "\n\n###\n\n"):
        super().__init__(datapath)
        self.end_prompt_token = end_prompt_token
        self.folder, self.file_name = os.path.split(self.datapath)

    def __call__(self):
        self.dataset = self._load_dataset_into_hf()

        self.dataset = self.dataset.map(self._convert_to_gpt, batched=False)
        self.dataset.remove_columns_(["input", "target"])

        print(self.datapath)
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
