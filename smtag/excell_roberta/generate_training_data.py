import argparse

from datasets import load_dataset

import glob
import logging
import os

from smtag.data_classes import TrainingExcellRoberta, TrainingArgumentsTOKCL

from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleLM
from .configuration_excell_roberta import EXcellRobertaConfig
from .modeling_excell_roberta import EXcellRobertaForMaskedLM

from transformers import (DataCollatorForLanguageModeling, HfArgumentParser,
                            RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM,
                            Trainer, DataCollatorForWholeWordMask)
logger = logging.getLogger('smtag.excell_roberta.model')



if __name__ == "__main__":

    # The first part of the code would be the argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="Path (absolute) or name to the tokenizer to be used to train the model")
    parser.add_argument("source_datap_ath", help="Path (absolute) to the data that will be used to train the model. train.jsonl and eval.jsonl are expected")
    parser.add_argument("target_data_path", help="Path (absolute) to the folder where the data will be stored")
    parser.add_argument("--max_length", default=512, type=int, help="Max length of the model")
    parser.add_argument("--block_size", default=256, type=int, choices=range(64,511), help="Size of the text blocks (in tokens) to generate examples")
    parser.add_argument( '-log',
                        '--loglevel',
                        default='warning',
                        help='Provide logging level. Example --loglevel debug, default=warning' ) 
    args = parser.parse_args()
    logging.basicConfig( level=args.loglevel.upper() )

    # The second part is defining the tokenizer, configuration, and model
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer, max_len=args.max_length)

    # The third part of the code is generating the data
    logger.info("Reading the dataset")

    ds = load_dataset("json", data_files={'train': os.path.join(args.source_datap_ath, "train.jsonl"),
                                          'eval': os.path.join(args.source_datap_ath, "eval.jsonl")})
    logger.info(print(ds))

    def tokenize_function(examples):
        try:
            return tokenizer(examples["body"])#, truncation=True, padding="max_length")
        except KeyError:
            return tokenizer(examples["text"])

    logger.info("Tokenizing datasets")
    tokenized_datasets = ds.map(tokenize_function, 
        batched=True, 
        # num_proc=64, 
        batch_size=1024,
        remove_columns=["text"])

    logger.info(ds["eval"][0])
    logger.info(tokenized_datasets["eval"][0])
    logger.info(tokenizer.decode(tokenized_datasets["eval"][0]["input_ids"]))

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Concatenating datasets with fix blocksize")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1024,
        # num_proc=64,
    )


# The part below i intended to generate a train and test partition
# This should generate an eval and a train set of the dataset. This should be later feeded to the training.

# # dataset is already `map`'d and already has `set_format`
# # 90% train, 10% test + validation
# train_testvalid = dataset.train_test_split(test_size=0.1)
# # Split the 10% test + valid in half test, half valid
# test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
# # gather everyone if you want to have a single DatasetDict
# datasets = DatasetDict({
#     "train": train_testvalid["train"],
#     "test": test_valid["test"],
#     "valid": test_valid["train"]})

# dataloaders = {partition: DataLoader(ds, batch_size=8) for partition, ds in datasets.items()}

# for batch in dataloaders["train"]:
#     print(batch.keys())
#     # dict_keys([])

    logger.info(ds["eval"][0])
    logger.info(lm_datasets["eval"][0])
    logger.info(tokenizer.decode(lm_datasets["eval"][0]["input_ids"]).replace("##",''))
    logger.info(tokenizer.convert_ids_to_tokens(lm_datasets["eval"][0]["input_ids"]))
    
    for split, dataset in lm_datasets.items():
        dataset.to_json(f"{args.target_data_path}/{split}.jsonl")