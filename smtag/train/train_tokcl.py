# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import NamedTuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    TrainingArguments, DataCollatorForTokenClassification,
    Trainer, IntervalStrategy,
    BartModel
)
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, GenerateMode
from ..models.experimental import (
    BecauseTokenClassification,
    BecauseConfigForTokenClassification,
)
from ..data_collator import DataCollatorForMaskedTokenClassification
from ..trainer import MyTrainer
from ..metrics import MetricsTOKCL
from ..show import ShowExampleTOCKL
from ..tb_callback import MyTensorBoardCallback
from ..config import config
from .. import LM_MODEL_PATH, TOKCL_MODEL_PATH, CACHE, RUNS_DIR


# changing default values
@dataclass
class TrainingArgumentsTOKCL(TrainingArguments):
    output_dir: str = field(default=TOKCL_MODEL_PATH)
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluation_strategy: str = field(default=IntervalStrategy.STEPS)
    prediction_loss_only: bool = field(default=True)  # crucial to avoid OOM at evaluation stage!
    learning_rate: float = field(default=1e-4)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: float = field(default=10.0)
    save_total_limit: int = field(default=5)
    masking_probability: float = field(default=None)
    replacement_probability: float = field(default=None)
    select_labels: bool = field(default=False)


def train(
    training_args: TrainingArgumentsTOKCL,
    loader_path: str,
    data_config_name: str,
    data_dir: str,
    no_cache: bool,
    tokenizer: AutoTokenizer = config.tokenizer,
    model_type: str = config.model_type,
    from_pretrained: str = LM_MODEL_PATH
):
    # copy training_args so that local modif don't affect subsequent training
    training_args = deepcopy(training_args)
    training_args.logging_dir = f"{RUNS_DIR}/tokcl-{data_config_name}-{datetime.now().isoformat().replace(':','-')}"
    output_dir = Path(training_args.output_dir) / data_config_name
    if not output_dir.exists():
        output_dir.mkdir()
        print(f"Created {output_dir}.")
    training_args.output_dir = str(output_dir)
    if (data_config_name == "NER"):
        # introduce noise to scramble entities to reinforce role of context over entity identity
        # make sure it is float even when zero!
        training_args.replacement_probability = 0.025 if training_args.replacement_probability is None else float(training_args.replacement_probability)
        # probabilistic masking
        training_args.masking_probability = 0.025 if training_args.masking_probability is None else float(training_args.masking_probability)
    elif data_config_name in ["GENEPROD_ROLES", "SMALL_MOL_ROLES"]:
        training_args.masking_probability = 1.0 if training_args.masking_probability is None else float(training_args.masking_probability)
        # pure contextual learning, all entities are masked
        training_args.replacement_probability = .0 if training_args.replacement_probability is None else float(training_args.replacement_probability)

    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading and tokenizing datasets found in {data_dir}.")
    print(f"using {loader_path} as dataset loader.")
    train_dataset, eval_dataset, test_dataset = load_dataset(
        path=loader_path,
        name=data_config_name,
        script_version="main",
        data_dir=data_dir,
        split=["train", "validation", "test"],
        download_mode=GenerateMode.FORCE_REDOWNLOAD if no_cache else GenerateMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=CACHE
    )
    print(f"\nTraining with {len(train_dataset)} examples.")
    print(f"Evaluating on {len(eval_dataset)} examples.")

    # if data_config_name in ["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES"]:
    # use our fancy data collator that randomly masks some of the inputs to enforce context learning
    training_args.remove_unused_columns = False  # we need tag_mask
    data_collator = DataCollatorForMaskedTokenClassification(
        tokenizer=tokenizer,
        # max_length=config.max_length,
        pad_to_multiple_of=config.max_length,
        masking_probability=training_args.masking_probability,
        replacement_probability=training_args.replacement_probability,
        select_labels=training_args.select_labels
        )
    # else:
    #     # normal token classification
    #     data_collator = DataCollatorForTokenClassification(
    #         tokenizer=tokenizer,
    #         max_length=config.max_length
    #     )

    num_labels = train_dataset.info.features['labels'].feature.num_classes
    label_list = train_dataset.info.features['labels'].feature.names
    print(f"\nTraining on {num_labels} features:")
    print(", ".join(label_list))

    compute_metrics = MetricsTOKCL(label_list=label_list)

    if model_type == "Autoencoder":
        model = AutoModelForTokenClassification.from_pretrained(
            from_pretrained,
            num_labels=num_labels,
            max_position_embeddings=config.max_length + 2  # max_length + 2 for start/end token
        )
        model_config = model.config
    elif model_type == "GraphRepresentation":
        # "The bare BART Model outputting raw hidden-states without any specific head on top."
        seq2seq = BartModel.from_pretrained(from_pretrained)  # use AutoModel instead? since LM head is provided by BecauseLM
        model_config = BecauseConfigForTokenClassification(
            freeze_pretrained='both',
            hidden_features=512,
            num_nodes=50,  # results into a num_nodes ** 2 latent var
            num_edge_features=6,  # not yet used
            num_node_features=10,
            sample_num_entities=20,
            sample_num_interactions=20,
            sample_num_interaction_types=3,
            sampling_iterations=100,
            alpha=1.,  # weight of adj_matrix_distro_loss
            beta=1.,  # weight of node_label_distro_loss
            gamma=0.,  # weight of the DAG loss
            seq_length=config.max_length,
            residuals=True,
            dropout=0.1,  # just to make it explicit
            classifier_dropout=0.1,
            num_labels=num_labels,
            max_position_embeddings=config.max_length + 2  # default is 1024
        )
        model = BecauseTokenClassification(
            pretrained=seq2seq,
            config=model_config
        )
    else:
        raise ValueError(f"unknown model type: {model_type}.")

    print(f"\nTraining arguments for model type {model_type}:")
    print(model_config)
    print(training_args)

    if model_type == "Autoencoder":
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[ShowExampleTOCKL(tokenizer)]
        )
    elif model_type == "GraphRepresentation":
        trainer = MyTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[ShowExampleTOCKL(tokenizer)]
        )
    else:
        raise ValueError(f"{model_type} is not implemented!")

    # switch the Tensorboard callback to plot losses on same plot
    trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
    trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback

    print(f"CUDA available: {torch.cuda.is_available()}")

    trainer.train() #ignore_keys_for_eval=['supp_data', 'adjascency', 'node_embeddings'])
    trainer.save_model(training_args.output_dir)

    print(f"Testing on {len(test_dataset)}.")
    trainer.args.prediction_loss_only = False
    pred: NamedTuple = trainer.predict(test_dataset, metric_key_prefix='test')
    print(f"{pred.metrics}")
