
from transformers import HfArgumentParser
from ...train.train_tokcl import TrainingArgumentsTOKCL, train
from ...config import config

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgumentsTOKCL), description="Traing script.")
    parser.add_argument("path", help="Path of the loader.")
    parser.add_argument("data_config_name", choices=["NER", "ROLES", "BORING", "PANELIZATION"], help="Name of the dataset configuration to use.")
    parser.add_argument("--data_dir", help="The dir for the dataset files to use for training.")
    parser.add_argument("--no_cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cache.")
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    path = args.path
    data_config_name = args.data_config_name
    data_dir = args.data_dir
    tokenizer = config.tokenizer  # tokenizer has to be the same application-wide
    train(
        no_cache,
        path,
        data_dir,
        data_config_name,
        training_args,
        tokenizer
    )
