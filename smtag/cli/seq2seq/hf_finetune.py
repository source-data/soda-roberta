from argparse import Action
from transformers import HfArgumentParser
from smtag.train.train_seq2seq import HfSeq2SeqTrainer
from smtag.data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq
import logging
import re
import pandas as pd

logger = logging.getLogger('soda-roberta.train_seq2seq.HfSeq2SeqTrainer')

def separate_labels(str_, labels):

    output_str = ""

    for label in labels:
        regex_string = fr"<{label}> (.*?) </{label}>"
        output_str += f"- {label}: {','.join(set(re.compile(regex_string).findall(str_)))} \n "

    output_str += "[END]"
    return output_str

if __name__ == "__main__":
    parser = HfArgumentParser([ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq], description="Traing script.")
    parser.add_argument("file_path", help="Path to the csv text file containing the data. It must follow the input#separator#output schema.")
    parser.add_argument("task", help="Choose between NER, PANEL, CAUSAL")
    parser.add_argument("task_type", help="Choose between copy_tag or list. Copy tag copies the sentence tagging the entities and list returns the found entities.")
    parser.add_argument("--delimiter", 
                        default="###tt9HHSlkWoUM###", 
                        type=str,
                        help="Delimiter used in the file to separate input and output.")
    parser.add_argument("--base_model", 
                        default="t5-base", 
                        type=str,
                        help="Model to be used for training. If local checkpoint provided, the base model to select the class.")
    parser.add_argument("--from_local_checkpoint", 
                        default="", 
                        type=str,
                        help="Local checkpoint to be used.")
    parser.add_argument("--prompt_init", 
                        default="Do NER on the entities",
                        type=str,
                        help="Task name to put at the beginning of the prompt for NER task. The name of the labels to be found will be generated in the code.")
    parser.add_argument("--prompt_end", 
                        default="\n\nEND_INPUT\n\n", 
                        type=str,
                        help="Tokend to indicate end of input and that the model must generate text.")
    parser.add_argument("--generate_end", 
                        default="[END]", 
                        type=str,
                        help="End of generation token.")
    parser.add_argument("--skip_lines", 
                        default=0, 
                        type=int,
                        help="First lines of the file to skip.")
    parser.add_argument("--split",
                        nargs=3, 
                        default=[0.8, 0.1, 0.1], 
                        type=float, 
                        help="Fraction of the dataset to be split into train, validation, test.")
    parser.add_argument("--max_input_length", 
                        default=512, 
                        type=int,
                        help="Maximum length accepted by the tokenizer as input.")
    parser.add_argument("--max_target_length", 
                        default=512, 
                        type=int,
                        help="Maximum length accepted by the tokenizer as output.")
    parser.add_argument("--ner_labels",
                        nargs="*", 
                        type=str,
                        default="all" ,
                        help="Which NER entities are to be classify. Choose all or any combination of: [GENEPROD, TISSUE, ORGANISM, SMALL_MOLECULE, EXP_ASSAY, CELL, SUBCELLULAR].")
                               
    model_config, trainer_config, args = parser.parse_args_into_dataclasses()    

    trainer = HfSeq2SeqTrainer(
                 # DATA AND MODELS
                 datapath=args.file_path,
                 task=args.task,
                 task_type=args.task_type,
                 labels_list=args.ner_labels,
                 delimiter=args.delimiter,
                 base_model=args.base_model,
                 from_local_checkpoint=args.from_local_checkpoint,
                 # SPECIAL FOR NER
                 prompt_init=args.prompt_init,
                 prompt_end=args.prompt_end,
                 generate_end=args.generate_end,
                 # DATA GENERATION
                 split=args.split,
                 skip_lines=args.skip_lines,
                 # TOKENIZER PARAMETERS
                 max_input_length=args.max_input_length,
                 max_target_length=args.max_target_length,
                 # MODEL PARAMETERS
                 model_param=model_config,
                 # TRAINING PARAMETERS
                 training_args=trainer_config
                 )

    trainer()






