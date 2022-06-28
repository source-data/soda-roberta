from transformers import TrainingArguments, IntervalStrategy
from dataclasses import dataclass, field
from typing import List, Optional
from smtag import TOKCL_MODEL_PATH


@dataclass
class TrainingArgumentsTOKCL(TrainingArguments):
    output_dir: str = field(default=TOKCL_MODEL_PATH)
    # Main Hyperparameters to tune
    learning_rate: float = field(default=1e-4)
    lr_schedule: str = field(default='linear')
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    num_train_epochs: float = field(default=1.)
    masking_probability: float = field(default=None)
    classifier_dropout: float = field(default=0.25)
    replacement_probability: float = field(default=None)
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.0)
    warmup_steps: int = field(default=0)

    # Logging and evaluation strategy
    evaluation_strategy: IntervalStrategy = field(default="steps")
    eval_steps: int = field(default=1000)
    prediction_loss_only: bool = field(default=False)
    eval_accumulation_steps: Optional[int] = field(default=None)
    log_level: Optional[str] = field(default="passive")
    logging_dir: Optional[str] = field(default=None)
    logging_strategy: IntervalStrategy = field(default="steps")
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=100)
    logging_nan_inf_filter: str = field(default=True)
    save_strategy: IntervalStrategy = field(default="steps")
    save_steps: int = field(default=1000)
    save_total_limit: Optional[int] = field(default=None)
    seed: int = field(default=42)
    select_labels: bool = field(default=False)

    # Optimization parameters
    gradient_accumulation_steps: int = field(default=1)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})

    # Folders and identifications
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=False)
    run_name: Optional[str] = field(default="Seq2Seq")

    # Other params 
    disable_tqdm: Optional[bool] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=True)
    metric_for_best_model: Optional[str] = field(default='f1')
    greater_is_better: Optional[bool] = field(default=True)
    report_to: Optional[List[str]] = field(default='tensorboard')
    resume_from_checkpoint: Optional[str] = field(default=None)

    # HuggingFace Hub parameters
    # push_to_hub: bool = field(default=False)
    # hub_model_id: str = field(default=None)
    # hub_strategy: HubStrategy = field(default="every_save")
    # hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    # gradient_checkpointing: bool = field(default=False)
