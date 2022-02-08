
from transformers import HfArgumentParser

from smtag import LM_MODEL_PATH
from ...train.train_tokcl import TrainingArgumentsTOKCL, train
from ...config import config

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgumentsTOKCL), description="Traing script.")
    parser.add_argument("loader_path", help="Path of the loader.")
    parser.add_argument("data_config_name", choices=["NER", "ROLES", "BORING", "PANELIZATION"], help="Name of the dataset configuration to use.")
    parser.add_argument("--data_dir", help="The dir for the dataset files to use for training.")
    parser.add_argument("--no_cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cache.")
    parser.add_argument("--from_pretrained", default=LM_MODEL_PATH, help="The pretrained model to fine tune.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    loader_path = args.loader_path
    data_config_name = args.data_config_name
    data_dir = args.data_dir
    from_pretrained = args.from_pretrained
    tokenizer = config.tokenizer  # tokenizer has to be the same application-wide
    train(
        training_args,
        loader_path,
        data_config_name,
        data_dir,
        no_cache,
        tokenizer,
        from_pretrained
    )
