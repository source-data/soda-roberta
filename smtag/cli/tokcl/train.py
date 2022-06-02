
from transformers import HfArgumentParser

from smtag import LM_MODEL_PATH
from ...train.train_tokcl_benchmark import TrainingArgumentsTOKCL, TrainModel
from ...config import config

if __name__ == "__main__":
    # parser = HfArgumentParser(TrainingArgumentsTOKCL, description="""Class defining the arguments
    #                     of the HuggingFace Trainer class to be used.""")
    # parser.add_argument("loader_path", help="""Path to the data to be loaded. It can be a path to a
    #                     file stored in your computer or the name of the dataset repository
    #                     in the HuggingFace Hub.""")
    # parser.add_argument("task",
    #                     choices=["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"],
    #                     help="""Define the task on which the model must be fine tuned.""")
    # parser.add_argument("--data_dir", help="""The dir for the dataset files to use for training.
    #                     This data is no relevant if the dataset loader defined is a path to
    #                     a HuggingFace Hub dataset.""")
    # parser.add_argument("--no_cache", action="store_true", help="""Flag that forces re-donwloading the
    #                     dataset rather than re-using it from the cache.""")
    # parser.add_argument("--from_pretrained",
    #                     default=config.from_pretrained,
    #                     help="""The pretrained model to fine tune.
    #                     In many cases the tokenizer will be associated to the same.
    #                     However, it is dvisable to define the tokenizer too""")
    # parser.add_argument("--model_type", default="Autoencoder", help="The pretrained model to fine tune.")
    # parser.add_argument("--masked_data_collator",
    #                     default="False",
    #                     help="""Whether to use a normal or masked data collator.
    #                     If set to true, a tag_mask will need to be generated.
    #                     The reason to use the masked_data_collator is to
    #                     introduce noise to scramble entities to reinforce role of
    #                     context over entity identity.""")
    # parser.add_argument("--hidden_size_multiple", default="50",
    #                     help="""Number of neurons in the classifier layer. It will be multiplied
    #                     by the number of transformer layers in the model. So for BERT,
    #                     a hidde_size_multiple of 50 would be multiplied by 12 and generate
    #                     a layer with 600 neurons.""")
    # parser.add_argument("--dropout",
    #                     default="0.2",
    #                     help="""Dropout rate for the classifier layers.""")
    # parser.add_argument("--tokenizer", default=None, help="The pretrained model to fine tune.")
    # parser.add_argument("--do_test",
    #                     default=True,
    #                     help="""If set to true, it will use the test dataset to do an
    #                     inference and return the performance of the model in the
    #                     test dataset.""")
    # parser.add_argument("--test_results_file",
    #                     default="./test_results_benchmark.json",
    #                     help="""JSON file with the results of the model.""")

    # tasks = ["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"]
    tasks = ["NER"]
    models = {
        "EMBO/bio-lm": "EMBO/bio-lm",
        "EMBO/bert-base-cased": "bert-base-cased",
        "EMBO/bert-base-uncased": "bert-base-uncased",
        "EMBO/bert-large-cased": "bert-large-cased",
        "EMBO/bert-large-uncased": "bert-large-uncased",
        "EMBO/roberta-base": "roberta-base",
        "EMBO/roberta-large": "roberta-large",
        "EMBO/biobert-base-cased": "dmis-lab/biobert-base-cased-v1.2",
        "EMBO/biobert-large-cased": "dmis-lab/biobert-large-cased-v1.1",
        "EMBO/PubMedBERT-base-uncased-abstract-fulltext": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "EMBO/BiomedNLP-KRISSBERT": "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
        "EMBO/PubMedBERT-base-uncased-abstract": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "EMBO/BioMegatron345mCased": "EMBO/BioMegatron345mCased",
        "EMBO/BioMegatron345mUncased": "EMBO/BioMegatron345mUncased"
    }

    learning_rate = [0.0001] # [0.0001, 0.00005]
    lr_scheduler = ["linear"] # ["constant", "linear", "cosine"]
    dropout_list = [0.1] # Dropout is very similar. We will leave 0.1
    per_device_train_batch_size = 16
    hidden_size_multiple_values = [68] # [32, 64] # 16 learns too slow and 128 does not even learn with masked
    prediction_loss_only = False
    evaluation_strategy = "epoch"
    eval_steps = 1
    save_total_limit = 500
    num_train_epochs = 12
    save_strategy = "epoch"
    save_steps = 500
    seed = 42
    load_best_model_at_end = True
    metric_best_model = "f1"
    greater_is_better = True
    report_to = "tensorboard"
    push_to_hub = False
    hub_strategy = "checkpoint"
    overwrite_output_dir = True
    test_results_file = "benchmarking_results.pkl"
    model_type = "Autoencoder"
    do_test = True

    for model in models.keys():
        dataset_to_load = "drAbreu/sd-nlp-2" if models[model] in ["EMBO/bio-lm",
                                                                  "roberta-base",
                                                                  "roberta-large"] else "EMBO/sd-nlp-non-tokenized"
        tokenizer_name = "roberta-base" if models[model] in ["EMBO/bio-lm"] else models[model]
        for task in tasks:
            for lr in learning_rate:
                for lr_schedule in lr_scheduler:
                    for masked in [True]: #[False, True]:
                        for dr in dropout_list:
                            for hidden_size in hidden_size_multiple_values:
                                hub_model_id = f"EMBO/{model}_{task}"
                                output_dir = f"./tokcl_models/EMBO_{model}_{task}"
                                test_results_file = "benchmarking_results.pkl"
                                # Here I should add the parameters for the base model.
                                # That would give a much better control over the benchmark
                                training_args = {
                                    "learning_rate": lr,
                                    "lr_scheduler_type": lr_schedule,
                                    "per_device_eval_batch_size": 64,
                                    "per_device_train_batch_size": per_device_train_batch_size,
                                    "prediction_loss_only": False,
                                    "evaluation_strategy": "epoch",
                                    "eval_steps": 1,
                                    "save_total_limit": 500,
                                    "num_train_epochs": num_train_epochs,
                                    "save_strategy": "epoch",
                                    "save_steps": 1,
                                    "seed": 42,
                                    "load_best_model_at_end": True,
                                    "metric_for_best_model": "f1",
                                    "greater_is_better": True,
                                    "report_to": "tensorboard",
                                    "push_to_hub": False,
                                    "hub_strategy": "checkpoint",
                                    "overwrite_output_dir": True
                                }

                                loader_path = dataset_to_load
                                task = task
                                from_pretrained = models[model]
                                dropout = dr
                                hidden_size_multiple = hidden_size
                                masked_data_collator = masked
                                print(f'data: {loader_path}, pre-trained: {from_pretrained}, tokenizer: {tokenizer_name}')

                                trainer = TrainModel(
                                    training_args=TrainingArgumentsTOKCL(**training_args),
                                    loader_path=loader_path,
                                    task=task,
                                    from_pretrained=from_pretrained,
                                    tokenizer_name=tokenizer_name,
                                    model_type=model_type,
                                    do_test=do_test,
                                    dropout=dropout,
                                    hidden_size_multiple=hidden_size_multiple,
                                    masked_data_collator=masked_data_collator,
                                    file_=test_results_file
                                )

                                trainer()

                                trainer.save_benchmark_results()