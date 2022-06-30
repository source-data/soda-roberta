
from transformers import HfArgumentParser

from smtag import LM_MODEL_PATH
from ...train.train_tokcl import TrainTokenClassification, HpSearchForTokenClassification
from smtag.data_classes import TrainingArgumentsTOKCL
from ...config import config
import logging
from ray import tune

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
    parser.add_argument("--masked_data_collator", 
                        action="store_true", 
                        help="Whether to randomly mask tokens in the data collator or not.")
    parser.add_argument("--hyperparameter_search", 
                        action="store_true", 
                        help="""Activates the hyperparameter search for the model.
                                The configuration for the hyperparameter tuning should be 
                                given as part of the config.py file.""")
    parser.add_argument("--smoke_test", 
                        action="store_true", 
                        help="""Runs a very fast version of the hyperparameter search.
                                Specially used for debugging or developing purposes.""")
    parser.add_argument("--hp_gpus_per_trial", 
                        default=1, 
                        help="Number of GPUs to use per each trial. Only for hp search")
    parser.add_argument("--hp_tune_samples", 
                        default=8, 
                        help="Total number of samples to generate for hp_search.")
    parser.add_argument("--hp_experiment_name", 
                        default="tune_transformer_pbt", 
                        help="Name for the experiment to be stored in the system.")
    parser.add_argument("--hp_local_folder", 
                        default="/app/data/ray_results/", 
                        help="Generic folder where the data will be stored.")
                               
    training_args, args = parser.parse_args_into_dataclasses()
    no_cache = args.no_cache
    loader_path = args.loader_path
    task = args.task
    data_dir = args.data_dir
    from_pretrained = args.from_pretrained
    model_type = config.model_type
    tokenizer = config.tokenizer  
    masked_data_collator = args.masked_data_collator  
    hyperparameter_search = args.hyperparameter_search  
    hp_gpus_per_trial = int(args.hp_gpus_per_trial)  
    hp_tune_samples = int(args.hp_tune_samples) 
    hp_experiment_name = args.hp_experiment_name  
    hp_local_folder = args.hp_local_folder  
    if hyperparameter_search:
        smoke_test = args.smoke_test  
        hp_search_config = config.hp_search_config  
        hp_search_config["max_steps"] = 1 if smoke_test else -1
        hp_search_scheduler = config.hp_search_scheduler  
        hp_search_reporter = config.hp_search_reporter  
        if ('large' in from_pretrained) or ('Megatron345m' in from_pretrained):
            hp_search_config["per_device_train_batch_size"] = tune.choice([4, 8, 16])
            hp_search_config["per_device_eval_batch_size"] = 32

        hp_search = HpSearchForTokenClassification(
            training_args=training_args,
            loader_path=loader_path,
            task=task,
            from_pretrained=from_pretrained,
            model_type=model_type,
            masked_data_collator=masked_data_collator,
            data_dir=data_dir,
            no_cache=no_cache,
            tokenizer=tokenizer,
            smoke_test=args.smoke_test,
            gpus_per_trial=hp_gpus_per_trial,
            hp_tune_samples=hp_tune_samples,
            hp_search_config=hp_search_config,
            hp_search_scheduler=hp_search_scheduler,
            hp_search_reporter=hp_search_reporter,
            hp_experiment_name=hp_experiment_name,
            hp_local_dir=hp_local_folder, 
        )

        best_model = hp_search._run_hyperparam_search()
    else:
        trainer = TrainTokenClassification(
            training_args=training_args,
            loader_path=loader_path,
            task=task,
            from_pretrained=from_pretrained,
            model_type=model_type,
            masked_data_collator=masked_data_collator,
            data_dir=data_dir,
            no_cache=no_cache,
            tokenizer=tokenizer,
        )

        trainer()
