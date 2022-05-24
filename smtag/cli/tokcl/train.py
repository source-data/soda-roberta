
from transformers import HfArgumentParser

from smtag import LM_MODEL_PATH
from ...train.train_tokcl_benchmark import TrainingArgumentsTOKCL, TrainModel
from ...config import config

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArgumentsTOKCL, description="""Class defining the arguments
                        of the HuggingFace Trainer class to be used.""")
    parser.add_argument("loader_path", help="""Path to the data to be loaded. It can be a path to a
                        file stored in your computer or the name of the dataset repository
                        in the HuggingFace Hub.""")
    parser.add_argument("task",
                        choices=["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"],
                        help="""Define the task on which the model must be fine tuned.""")
    parser.add_argument("--data_dir", help="""The dir for the dataset files to use for training.
                        This data is no relevant if the dataset loader defined is a path to 
                        a HuggingFace Hub dataset.""")
    parser.add_argument("--no_cache", action="store_true", help="""Flag that forces re-donwloading the 
                        dataset rather than re-using it from the cache.""")
    parser.add_argument("--from_pretrained",
                        default=config.from_pretrained,
                        help="""The pretrained model to fine tune.
                        In many cases the tokenizer will be associated to the same.
                        However, it is dvisable to define the tokenizer too""")
    parser.add_argument("--model_type", default="Autoencoder", help="The pretrained model to fine tune.")
    parser.add_argument("--masked_data_collator",
                        default="False",
                        help="""Whether to use a normal or masked data collator.
                        If set to true, a tag_mask will need to be generated.
                        The reason to use the masked_data_collator is to 
                        introduce noise to scramble entities to reinforce role of 
                        context over entity identity.""")
    parser.add_argument("--hidden_size_multiple", default="50",
                        help="""Number of neurons in the classifier layer. It will be multiplied
                        by the number of transformer layers in the model. So for BERT, 
                        a hidde_size_multiple of 50 would be multiplied by 12 and generate
                        a layer with 600 neurons.""")
    parser.add_argument("--dropout",
                        default="0.2",
                        help="""Dropout rate for the classifier layers.""")
    parser.add_argument("--tokenizer", default=None, help="The pretrained model to fine tune.")
    parser.add_argument("--do_test",
                        default=True,
                        help="""If set to true, it will use the test dataset to do an
                        inference and return the performance of the model in the 
                        test dataset.""")
    parser.add_argument("--test_results_file",
                        default="./test_results_benchmark.json",
                        help="""JSON file with the results of the model.""")

    # Here I should add the parameters for the base model.
    # That would give a much better control over the benchmark
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    loader_path = args.loader_path
    task = args.task
    data_dir = args.data_dir
    from_pretrained = args.from_pretrained
    model_type = args.model_type
    dropout = float(args.dropout)
    hidden_size_multiple = int(args.hidden_size_multiple)
    masked_data_collator = bool(args.masked_data_collator)
    do_test = bool(args.do_test)
    tokenizer_name = args.tokenizer if args.tokenizer else from_pretrained
    test_results_file = args.test_results_file

    trainer = TrainModel(
        training_args=training_args,
        loader_path=loader_path,
        task=task,
        from_pretrained=from_pretrained,
        tokenizer_name=tokenizer_name,
        model_type=model_type,
        do_test=do_test,
        data_dir=data_dir,
        dropout=dropout,
        hidden_size_multiple=hidden_size_multiple,
        no_cache=no_cache,
        file_= test_results_file
    )

    trainer()

    trainer.save_benchmark_results()