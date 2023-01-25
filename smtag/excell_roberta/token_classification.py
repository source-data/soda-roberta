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
                            RobertaTokenizerFast, Trainer, AutoTokenizer)
from typing import Union, Tuple
from datasets import DatasetDict
from smtag.xml2labels import SourceDataCodes as sd
from smtag.excell_roberta.dataprep import PreparatorTOKCL

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


CODE_MAPS = {
    "NER": sd.ENTITY_TYPES, 
    "GENEPROD_ROLES": sd.GENEPROD_ROLES, 
    "SMALL_MOL_ROLES": sd.SMALL_MOL_ROLES, 
    "PANELIZATION": sd.PANELIZATION
}

ID2LABEL = {
    "NER": {
        "0": "O",
        "1": "B-SMALL_MOLECULE",
        "2": "I-SMALL_MOLECULE",
        "3": "B-GENEPROD",
        "4": "I-GENEPROD",
        "5": "B-SUBCELLULAR",
        "6": "I-SUBCELLULAR",
        "7": "B-CELL",
        "8": "I-CELL",
        "9": "B-TISSUE",
        "10": "I-TISSUE",
        "11": "B-ORGANISM",
        "12": "I-ORGANISM",
        "13": "B-EXP_ASSAY",
        "14": "I-EXP_ASSAY",
        "15": "B-DISEASE",
        "16": "I-DISEASE"        
    }, 
    "GENEPROD_ROLES": {
        "0": "O",
        "1": "B-CONTROLLED_VAR",
        "2": "I-CONTROLLED_VAR",
        "3": "B-MEASURED_VAR",
        "4": "I-MEASURED_VAR"
    }, 
    "SMALL_MOL_ROLES": {
        "0": "O",
        "1": "B-CONTROLLED_VAR",
        "2": "I-CONTROLLED_VAR",
        "3": "B-MEASURED_VAR",
        "4": "I-MEASURED_VAR"
    }, 
    "PANELIZATION": {
        "0": "O",
        "1": "B-PANEL_START",
        "2": "I-PANEL_START"
    }

}

if __name__ == "__main__":

    # The first part of the code would be the argument parsing
    parser = HfArgumentParser(TrainingArgumentsTOKCL, description="Traing script.")
    # parser = argparse.ArgumentParser()
    parser.add_argument("data", 
        help="Path to directory containing the data"
        )
    parser.add_argument("--task",
        default="NER",
        # choices=["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"], 
        help="The task for which we want to train the token classification."
        )
    parser.add_argument("--tokenizer", 
        default="",
        help="Path (absolute) or name to the tokenizer to be used to train the model",
        )
    parser.add_argument("--model", 
        default="",
        help="Path (absolute) or name to the model",
        )
    parser.add_argument("--crf", 
        action="store_true", 
        help="If true, it adds a CRF layer to the classifier."
        )
    parser.add_argument("--include_start_end_transitions", 
        action="store_true", 
        help="If true, it adds a CRF layer to the classifier."
        )
    parser.add_argument("--max_length",
        type=int,
        default=512 ,
        help="""Maximum length accepted by the model."""
        )
    parser.add_argument("--masked_data_collator", 
                        action="store_true", 
                        help="Whether to randomly mask tokens in the data collator or not.")

    training_args, args = parser.parse_args_into_dataclasses()
    # logging.basicConfig( level=args.loglevel.upper() )
    code_map = CODE_MAPS[args.task] 
    source_dir_path = args.data
    id2label = ID2LABEL[args.task]
    label2id = {v: k for k, v in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, 
        add_prefix_space=True
        )


    logging.info("""Reading the data""")
    
    # data = load_dataset(args.data, args.task)

    # if "tokens" in data["train"].column_names:
    #     data = DatasetDict(
    #         {
    #             "train": data["train"].rename_columns({"tokens": "words", "ner_tags": "labels"}),
    #             "validation": data["validation"].rename_columns({"tokens": "words", "ner_tags": "labels"}),
    #             "test": data["test"].rename_columns({"tokens": "words", "ner_tags": "labels"})
    #         }
    #     )
    # id2label, label2id = _get_data_labels(data["train"])
    
    # if args.ner_labels == "all":
    #     ignore_keys = None
    # else:
    #     labels_not_bio = ['GENEPROD', 'TISSUE', 'ORGANISM', 'SMALL_MOLECULE', 'EXP_ASSAY', 'CELL', 'SUBCELLULAR']
    #     ignore_keys = [label for label in labels_not_bio if label not in args.ner_labels]
    dataprep = PreparatorTOKCL(args.data, code_map, args.tokenizer, max_length=args.max_length)
    data = dataprep.run()

    logging.info("""Loading the tokenizer""")
    if args.crf:
        model = EXcellRobertaForTokenClassification.from_pretrained(
            args.model, 
            use_crf=args.crf, 
            num_labels=len(list(label2id.keys())),
            id2label=id2label,
            label2id=label2id,
            return_dict=False,
            include_start_end_transitions=args.include_start_end_transitions,
            max_length=args.max_length
            )
        compute_metrics = MetricsCRFTOKCL(label_list=list(label2id.keys()))
    else:
        compute_metrics = MetricsTOKCL(label_list=list(label2id.keys()))
        model = EXcellRobertaForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(list(label2id.keys())),
            id2label=id2label,
            label2id=label2id,
            max_length=args.max_length,
            return_dict=False,
           )

    logging.info("""Loading the model""")
    # tokenized_data = data.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     remove_columns=data['train'].column_names
    #     )

    logger.info("Instantiating DataCollatorForTokenClassification")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                               return_tensors='pt',
                                                               padding="longest",
                                                               max_length=model.config.max_length)

    
    logger.info("Training the model")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        compute_metrics=compute_metrics,
        callbacks=[MyTensorBoardCallback]
    )

    trainer.train()

    logger.info("Evaluating the model in the test set")
    trainer.args.prediction_loss_only = False
    pred = trainer.predict(
        data["test"], 
        metric_key_prefix='test')#,
#        ignore_keys=ignore_keys)
