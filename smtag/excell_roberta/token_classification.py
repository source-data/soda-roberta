from datasets import load_dataset, Dataset
import logging
import os

from smtag.data_classes import TrainingArgumentsTOKCL
from smtag.metrics import MetricsTOKCL, MetricsCRFTOKCL

from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleTOKCL
from .modeling_excell_roberta import EXcellRobertaForTokenClassification
from .configuration_excell_roberta import EXcellRobertaConfig

from transformers import (HfArgumentParser, DataCollatorForTokenClassification,
                            RobertaTokenizerFast, Trainer)
from typing import Union, Tuple
from datasets import DatasetDict
logger = logging.getLogger('smtag.excell_roberta.model')

def shift_label(label):
    # If the label is B-XXX we change it to I-XXX
    if label % 2 == 1:
        label += 1
    return label

def tokenize_sentences(example):
    return tokenizer(example["words"], 
                     is_split_into_words=True, 
                     return_tensors="pt")

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

def _get_data_labels(data: Dataset) -> Tuple[dict, dict]:
    num_labels = data.info.features['labels'].feature.num_classes
    label_list = data.info.features['labels'].feature.names
    id2label, label2id = {}, {}
    for class_, label in zip(range(num_labels), label_list):
        id2label[class_] = label
        label2id[label] = class_
    logging.info(f"The data set has {num_labels} features: {label_list}")
    return id2label, label2id

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['words'], 
        truncation=True,
        is_split_into_words=True,
        max_length=model.config.max_length
        )
    
    all_labels = examples['labels']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
        
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

if __name__ == "__main__":

    # The first part of the code would be the argument parsing
    parser = HfArgumentParser(TrainingArgumentsTOKCL, description="Traing script.")
    # parser = argparse.ArgumentParser()
    parser.add_argument("--data", 
        default="EMBO/sd-nlp-non-tokenized", 
        help="Path (absolute) or name to the tokenizer to be used to train the model"
        )
    parser.add_argument("--model", 
        default="/app/excell-roberta-lm",
        help="Path (absolute) or name to the tokenizer to be used to train the model",
        )
    parser.add_argument("--task",
        default="NER",
        # choices=["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"], 
        help="The task for which we want to train the token classification."
        )
    parser.add_argument("--add_prefix_space", 
        action="store_true", 
        help="Set to true if uing roberta with word splitted lists."
        )
    parser.add_argument("--crf", 
        action="store_true", 
        help="If true, it adds a CRF layer to the classifier."
        )
    parser.add_argument("--include_start_end_transitions", 
        action="store_true", 
        help="If true, it adds a CRF layer to the classifier."
        )
    parser.add_argument("--ner_labels",
        nargs="*", 
        type=str,
        default="all" ,
        help="""Which NER entities are to be classify. Choose all or any combination of: 
        [GENEPROD, TISSUE, ORGANISM, SMALL_MOLECULE, EXP_ASSAY, CELL, SUBCELLULAR]."""
        )
    parser.add_argument("--masked_data_collator", 
                        action="store_true", 
                        help="Whether to randomly mask tokens in the data collator or not.")

   

    training_args, args = parser.parse_args_into_dataclasses()
    # logging.basicConfig( level=args.loglevel.upper() )

    logging.info("""Reading the data""")
    
    data = load_dataset(args.data, args.task)

    if "tokens" in data["train"].column_names:
        data = DatasetDict(
            {
                "train": data["train"].rename_columns({"tokens": "words", "ner_tags": "labels"}),
                "validation": data["validation"].rename_columns({"tokens": "words", "ner_tags": "labels"}),
                "test": data["test"].rename_columns({"tokens": "words", "ner_tags": "labels"})
            }
        )
    id2label, label2id = _get_data_labels(data["train"])
    
    if args.ner_labels == "all":
        ignore_keys = None
    else:
        labels_not_bio = ['GENEPROD', 'TISSUE', 'ORGANISM', 'SMALL_MOLECULE', 'EXP_ASSAY', 'CELL', 'SUBCELLULAR']
        ignore_keys = [label for label in labels_not_bio if label not in args.ner_labels]
    
    logging.info("""Loading the tokenizer""")
    config = EXcellRobertaConfig.from_pretrained(args.model,
                num_labels=len(list(label2id.keys())),
                id2label=id2label,
                label2id=label2id,
                return_dict=False,
                )
    if args.crf:
        model = EXcellRobertaForTokenClassification.from_pretrained(
            args.model, 
            use_crf=args.crf, 
            num_labels=len(list(label2id.keys())),
            id2label=id2label,
            label2id=label2id,
            return_dict=False,
            include_start_end_transitions=args.include_start_end_transitions
            )
        compute_metrics = MetricsCRFTOKCL(label_list=list(label2id.keys()))
    else:
        compute_metrics = MetricsTOKCL(label_list=list(label2id.keys()))
        model = EXcellRobertaForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(list(label2id.keys())),
            id2label=id2label,
            label2id=label2id,
            return_dict=False,
           )

    logging.info("""Loading the model""")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model, 
        is_pretokenized=True, 
        add_prefix_space=args.add_prefix_space
        )
    tokenized_data = data.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=data['train'].column_names
        )

    logger.info("Instantiating DataCollatorForTokenClassification")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                               return_tensors='pt',
                                                               padding="max_length",
                                                               max_length=model.config.max_length)

    
    logger.info("Training the model")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        compute_metrics=compute_metrics,
        callbacks=[MyTensorBoardCallback]
    )

    trainer.train()

    logger.info("Evaluating the model in the test set")
    trainer.args.prediction_loss_only = False
    pred = trainer.predict(
        tokenized_data["test"], 
        metric_key_prefix='test',
        ignore_keys=ignore_keys)
    import pandas as pd
    import numpy as np
    test_dataset = tokenized_data["test"]
    test_dataset.to_csv("./test_dataset.csv", sep="##delimiter##")
    pd.DataFrame({"predictions": np.array(pred.label_ids)}).to_csv("token_classification_predictions")
    