
from transformers import HfArgumentParser

from smtag import LM_MODEL_PATH
from ...train.train_tokcl import TrainTokenClassification
from smtag.data_classes import TrainingArgumentsTOKCL
from ...config import config
import logging

logger = logging.getLogger('soda-roberta.trainer.TOKCL')

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArgumentsTOKCL, description="Traing script.")
    parser.add_argument("--loader_path", 
                        help="Path of the loader.")
    parser.add_argument("--task", 
                        choices=["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"], 
                        help="The task for which we want to train the token classification.")
    parser.add_argument("--data_dir", 
                        help="The dir for the dataset files to use for training. Only needed if local data must be loaded.")
    parser.add_argument("--no_cache", 
                        action="store_true", 
                        help="Flag that forces re-donwloading the dataset rather than re-using it from the cache.")
    parser.add_argument("--from_pretrained", 
                        default=LM_MODEL_PATH, 
                        help="The pretrained model to fine tune.")
    parser.add_argument("--model_type", 
                        default="Autoencoder", 
                        help="The type of model to be used for training.")
    parser.add_argument("--masked_data_collator", 
                        action="store_true", 
                        help="Whether to randomly mask tokens in the data collator or not.")

    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    loader_path = args.loader_path
    task = args.task
    data_dir = args.data_dir
    from_pretrained = args.from_pretrained
    model_type = args.model_type
    tokenizer = config.tokenizer  # tokenizer has to be the same application-wide
    masked_data_collator = args.masked_data_collator  # tokenizer has to be the same application-wide

    trainer = TrainTokenClassification(
        training_args,
        loader_path,
        task,
        from_pretrained,
        model_type,
        masked_data_collator,
        data_dir,
        no_cache,
        tokenizer,
    )

    trainer()
