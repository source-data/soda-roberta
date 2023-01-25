from transformers import TrainingArguments, IntervalStrategy, PretrainedConfig, Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import List, Optional
from smtag import TOKCL_MODEL_PATH, SEQ2SEQ_MODEL_PATH, LM_MODEL_PATH

@dataclass
class TrainingArgumentsTOKCL(TrainingArguments):
    output_dir: str = field(default=f"{TOKCL_MODEL_PATH}/excell-roberta-fine-tuned")
    # Main Hyperparameters to tune
    learning_rate: float = field(default=5e-5)
    lr_schedule: str = field(default='cosine')
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    num_train_epochs: float = field(default=3.)
    masking_probability: float = field(default=1.0)
    classifier_dropout: float = field(default=0.15)
    replacement_probability: float = field(default=0.0)
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.0)
    warmup_steps: int = field(default=0)

    # Logging and evaluation strategy
    evaluation_strategy: IntervalStrategy = field(default="epoch")
    eval_steps: int = field(default=0.5)
    prediction_loss_only: bool = field(default=False)
    eval_accumulation_steps: Optional[int] = field(default=None)
    log_level: Optional[str] = field(default="passive")
    logging_dir: Optional[str] = field(default=None)
    logging_strategy: IntervalStrategy = field(default="steps")
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=100)
    logging_nan_inf_filter: str = field(default=True)
    log_on_each_node: str = field(default=True)
    save_strategy: IntervalStrategy = field(default="epoch")
    save_steps: int = field(default=1)
    save_total_limit: Optional[int] = field(default=None)
    seed: int = field(default=42)
    select_labels: bool = field(default=False)

    # Optimization parameters
    gradient_accumulation_steps: int = field(default=1)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-10)
    max_grad_norm: float = field(default=1.0)
    adafactor: bool = field(default=True, metadata={"help": "Whether or not to replace AdamW by Adafactor."})

    # Folders and identifications
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    do_predict: bool = field(default=True)
    run_name: Optional[str] = field(default="excell-roberta-fine-tuned")

    # Other params 
    disable_tqdm: Optional[bool] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=True)
    metric_for_best_model: Optional[str] = field(default='f1')
    greater_is_better: Optional[bool] = field(default=True)
    report_to: Optional[List[str]] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    class_weights: Optional[bool] = field(default=False)

    # HuggingFace Hub parameters
    # push_to_hub: bool = field(default=False)
    # hub_model_id: str = field(default=None)
    # hub_strategy: HubStrategy = field(default="every_save")
    # hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    # gradient_checkpointing: bool = field(default=False)

@dataclass
class ModelConfigSeq2Seq(PretrainedConfig):
    max_length: int = field(default=128)
    min_length: int = field(default=0)
    do_sample: bool = field(default=True)
    early_stopping: bool = field(default=True)
    num_beams: int = field(default=1)
    num_beam_groups: int = field(default=1)
    diversity_penalty: float = field(default=0.0)
    temperature: float = field(default=1.0)
    top_k: int = field(default=20)
    top_p: float = field(default=0.90)
    num_return_sequences: int = field(default=1)
    output_scores: bool = field(default=False)
    length_penalty: float = field(default=50.)
    repetition_penalty: float = field(default=2.)
    no_repeat_ngram_size: int = field(default=5)
    return_dict_in_generate: bool = field(default=False)

@dataclass
class TrainingArgumentsSeq2Seq(Seq2SeqTrainingArguments):
    output_dir: str = field(default=SEQ2SEQ_MODEL_PATH)
    run_name: Optional[str] = field(default="Seq2Seq")
    generation_max_length: int = field(default=512)
    predict_with_generate: bool = field(default=True)
    generation_num_beams: int = field(default=None)

@dataclass
class Gpt3ModelParam:
    model: str = field(default="text-davinci-002")
    suffix: str = field(default=None)
    temperature: float = field(default=0.8)
    max_tokens: int = field(default=1024)
    top_p: float = field(default=1)
    n: int = field(default=1)
    stream: bool = field(default=False)
    echo: bool = field(default=False)
    frequency_penalty: float = field(default=0)
    presence_penalty: float = field(default=0)
    stop: str = field(default="[END]")
    best_of: int = 1
    logit_bias: dict = field(default=None)


class TrainingExcellRoberta(TrainingArguments):
    output_dir: str = field(default=LM_MODEL_PATH)
    # Main Hyperparameters to tune
    learning_rate: float = field(default=5e-5)
    lr_schedule: str = field(default='constant')
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    num_train_epochs: float = field(default=1.)
    classifier_dropout: float = field(default=0.25)
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.0)
    warmup_steps: int = field(default=0)

    # Logging and evaluation strategy
    evaluation_strategy: IntervalStrategy = field(default="steps")
    eval_steps: int = field(default=10000)
    prediction_loss_only: bool = field(default=False)
    eval_accumulation_steps: Optional[int] = field(default=None)
    log_level: Optional[str] = field(default="passive")
    logging_dir: Optional[str] = field(default=None)
    logging_strategy: IntervalStrategy = field(default="steps")
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=100)
    logging_nan_inf_filter: str = field(default=True)
    save_strategy: IntervalStrategy = field(default="steps")
    save_steps: int = field(default=50000)
    save_total_limit: Optional[int] = field(default=3)
    seed: int = field(default=42)
    select_labels: bool = field(default=False)

    # Optimization parameters
    gradient_accumulation_steps: int = field(default=1)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-10)
    max_grad_norm: float = field(default=1.0)
    adafactor: bool = field(default=True, metadata={"help": "Whether or not to replace AdamW by Adafactor."})

    # Folders and identifications
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    do_predict: bool = field(default=False)
    run_name: Optional[str] = field(default="excell-roberta")

    # Other params 
    disable_tqdm: Optional[bool] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=True)
    report_to: Optional[List[str]] = field(default='wandb')
    resume_from_checkpoint: Optional[str] = field(default=None)
