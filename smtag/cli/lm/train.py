from transformers import HfArgumentParser
from ...train.train_lm import TrainingArgumentsLM, train
from ...config import config

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgumentsLM), description="Traing script.")
    parser.add_argument("loader_path", help="Path of the loader.")
    parser.add_argument("data_config_name", nargs="?", default="MLM", choices=["MLM", "DET", "VERB", "SMALL", "NOUN", "SEQ2SEQ", "NOLM", "ROLES"], help="Name of the dataset configuration to use.")
    parser.add_argument("--data_dir", help="The dir for the dataset files to use for training.")
    parser.add_argument("--no_cache", action="store_true", help="Flag that forces re-donwloading the dataset rather than re-using it from the cache.")
    parser.add_argument("--from_pretrained", default=config.from_pretrained, help="The pretrained model to fine tune.")
    parser.add_argument("--model_type", default=config.model_type, help="The pretrained model to fine tune.")

    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    loader_path = args.loader_path
    data_config_name = args.data_config_name
    data_dir = args.data_dir
    from_pretrained = args.from_pretrained
    model_type = args.model_type
    tokenizer = config.tokenizer  # tokenizer has to be the same application-wide
    train(
        training_args,
        loader_path,
        data_config_name,
        data_dir,
        no_cache,
        tokenizer,
        model_type,
        from_pretrained
    )
