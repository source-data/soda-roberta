from transformers import HfArgumentParser
from smtag.train.train_seq2seq import HfSeq2SeqTrainer
# from smtag import LM_MODEL_PATH
# from ...train.train_tokcl import TrainTokenClassification, HpSearchForTokenClassification
from smtag.data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq
# from ...config import config
import logging
# from ray import tune
# from ray.tune.schedulers import PopulationBasedTraining, pbt

logger = logging.getLogger('soda-roberta.train_seq2seq.HfSeq2SeqTrainer')

if __name__ == "__main__":
    parser = HfArgumentParser([ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq], description="Traing script.")
    parser.add_argument("file_path", help="Path to the csv text file containing the data. It must follow the input#separator#output schema.")
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
                        type=float, 
                        help="Maximum length accepted by the tokenizer as output.")
                               
    model_config, trainer_config, args = parser.parse_args_into_dataclasses()
    
    trainer = HfSeq2SeqTrainer(
                 # DATA AND MODELS
                 datapath=args.file_path,
                 delimiter=args.delimiter,
                 base_model=args.base_model,
                 from_local_checkpoint=args.from_local_checkpoint,
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

