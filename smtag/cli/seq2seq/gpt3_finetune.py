from transformers import HfArgumentParser
from smtag.train.train_seq2seq import Gpt3FineTuner
from smtag.data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq
import logging

logger = logging.getLogger('soda-roberta.train_seq2seq.Gpt3FileConverter')

if __name__ == "__main__":
    parser = HfArgumentParser([ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq], description="Traing script.")
    parser.add_argument("file_path", help="Path to the csv text file containing the data. It must follow the input#separator#output schema.")
                               
    model_config, trainer_config, args = parser.parse_args_into_dataclasses()
    
    gpt = Gpt3FineTuner(datapath=args.file_path)

    ds =  gpt()
    print(ds["train"][0])

