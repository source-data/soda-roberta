python -m smtag.cli.tokcl.train EMBO/sd-nlp-non-tokenized NER \
                                --from_pretrained bert-base-cased \
                                --model_type Autoencoder \
                                --masked_data_collator True \
                                --tokenizer bert-base-cased \
                                --do_predict True \
                                --do_test True \
                                --dropout 0.0 \
                                --hidden_size_multiple 64 \
                                --prediction_loss_only False \
                                --per_device_train_batch_size 16\
                                --per_device_eval_batch_size 16\
                                --evaluation_strategy epoch \
                                --eval_steps 1 \
                                --label_smoothing_factor 0.0 \
                                --learning_rate 0.0001 \
                                --num_train_epochs 1 \
                                --lr_scheduler_type sinusoidal \
                                --num_warmup_steps 50 \
                                --save_strategy epoch \
                                --save_steps 1 \
                                --seed 42 \
                                --load_best_model_at_end True \
                                --metric_for_best_model f1 \
                                --greater_is_better True \
                                --report_to tensorboard \
                                --push_to_hub False \
                                --hub_strategy checkpoint \
                                --hub_model_id EMBO/bert-base-cased_NER-task \
                                --hub_token ${HUB_TOKEN} \
                                --adafactor False \
                                --adam_beta1 0.9 \
                                --adam_beta2 0.999 \
                                --adam_epsilon 1e-08 \
                                --weight_decay 0.0 \
                                --overwrite_output_dir \
                                --output_dir ./tokcl_models/EMBO_bert-base-cased_NER-task \
                                --test_results_file ./benchmarking_results.json





root@9288e76c4417:/app# python -m smtag.cli.tokcl.train EMBO/sd-nlp-non-tokenized NER \
>                                 --from_pretrained bert-base-cased \
>                                 --model_type Autoencoder \
>                                 --masked_data_collator True \
>                                 --tokenizer bert-base-cased \
>                                 --do_predict True \
>                                 --prediction_loss_only False \
>                                 --per_device_train_batch_size 16\
>                                 --per_device_eval_batch_size 16\
>                                 --evaluation_strategy steps \
>                                 --eval_steps 100 \
>                                 --label_smoothing_factor 0.0 \
>                                 --learning_rate 0.0001 \
>                                 --num_train_epochs 5 \
>                                 --lr_scheduler_type linear \
>                                 --save_strategy steps \
>                                 --save_steps 500 \
>                                 --seed 42 \
>                                 --load_best_model_at_end True \
>                                 --metric_for_best_model f1 \
>                                 --greater_is_better True \
>                                 --report_to tensorboard \
>                                 --push_to_hub True \
>                                 --hub_strategy checkpoint \
>                                 --hub_model_id EMBO/bert-base-cased_NER-task \
>                                 --hub_token ${HUB_TOKEN} \
>                                 --adafactor False \
>                                 --adam_beta1 0.9 \
>                                 --adam_beta2 0.999 \
>                                 --adam_epsilon 1e-08 \
>                                 --weight_decay 0.0 \
>                                 --overwrite_output_dir \
>                                 --output_dir ./tokcl_models/EMBO_bert-base-cased_NER-task


BertConfig {
  "_name_or_path": "bert-base-cased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "O",
    "1": "I-SMALL_MOLECULE",
    "2": "B-SMALL_MOLECULE",
    "3": "I-GENEPROD",
    "4": "B-GENEPROD",
    "5": "I-SUBCELLULAR",
    "6": "B-SUBCELLULAR",
    "7": "I-CELL",
    "8": "B-CELL",
    "9": "I-TISSUE",
    "10": "B-TISSUE",
    "11": "I-ORGANISM",
    "12": "B-ORGANISM",
    "13": "I-EXP_ASSAY",
    "14": "B-EXP_ASSAY"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "B-CELL": 8,
    "B-EXP_ASSAY": 14,
    "B-GENEPROD": 4,
    "B-ORGANISM": 12,
    "B-SMALL_MOLECULE": 2,
    "B-SUBCELLULAR": 6,
    "B-TISSUE": 10,
    "I-CELL": 7,
    "I-EXP_ASSAY": 13,
    "I-GENEPROD": 3,
    "I-ORGANISM": 11,
    "I-SMALL_MOLECULE": 1,
    "I-SUBCELLULAR": 5,
    "I-TISSUE": 9,
    "O": 0
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.15.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}

TrainingArgumentsTOKCL(
_n_gpu=4,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
bf16=False,
bf16_full_eval=False,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=True,
do_train=False,
eval_accumulation_steps=None,
eval_steps=100,
evaluation_strategy=IntervalStrategy.STEPS,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=True,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=EMBO/bert-base-cased_NER-task,
hub_strategy=HubStrategy.CHECKPOINT,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=0.0001,
length_column_name=length,
load_best_model_at_end=True,
local_rank=-1,
log_level=-1,
log_level_replica=-1,
log_on_each_node=True,
logging_dir=./tokcl_models/EMBO_bert-base-cased_NER-task/runs/May20_12-20-10_9288e76c4417,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=50,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_type=SchedulerType.LINEAR,
masking_probability=None,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=f1,
mp_parameters=,
no_cuda=False,
num_train_epochs=5.0,
output_dir=./tokcl_models/EMBO_bert-base-cased_NER-task,
overwrite_output_dir=True,
past_index=-1,
per_device_eval_batch_size=16,
per_device_train_batch_size=16,
prediction_loss_only=False,
push_to_hub=True,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
remove_unused_columns=True,
replacement_probability=None,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=./tokcl_models/EMBO_bert-base-cased_NER-task,
save_on_each_node=False,
save_steps=500,
save_strategy=IntervalStrategy.STEPS,
save_total_limit=5,
seed=42,
select_labels=False,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_legacy_prediction_loop=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
xpu_backend=None,
)
/app/./tokcl_models/EMBO_bert-base-cased_NER-task is already a clone of https://huggingface.co/EMBO/bert-base-cased_NER-task. Make sure you pull the latest changes with `repo.git_pull()`.
WARNING:/app/./tokcl_models/EMBO_bert-base-cased_NER-task is already a clone of https://huggingface.co/EMBO/bert-base-cased_NER-task. Make sure you pull the latest changes with `repo.git_pull()`.
***** Running training *****
  Num examples = 48771
  Num Epochs = 5
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 1
  Total optimization steps = 3815
  0%|                                                                                                                                  | 0/3815 [00:00<?, ?it/s]/opt/conda/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.6483, 'learning_rate': 9.868938401048493e-05, 'epoch': 0.07}
{'loss': 0.3433, 'learning_rate': 9.737876802096986e-05, 'epoch': 0.13}
  3%|███▏                                                                                                                    | 100/3815 [00:52<27:37,  2.24it/s]***** Running Evaluation *****
  Num examples = 13801
  Batch size = 64

                                                                                ██████████████████████████████████████████████| 216/216 [00:40<00:00,  6.54it/s]


                precision    recall  f1-score   support

          CELL       0.71      0.75      0.73     27474
     EXP_ASSAY       0.67      0.68      0.67     54365
      GENEPROD       0.74      0.88      0.80    132086
      ORGANISM       0.70      0.66      0.68     13041
SMALL_MOLECULE       0.72      0.58      0.64     37519
   SUBCELLULAR       0.61      0.62      0.62     18836
        TISSUE       0.61      0.54      0.57     13344

     micro avg       0.71      0.75      0.73    296665
     macro avg       0.68      0.67      0.67    296665
  weighted avg       0.71      0.75      0.72    296665

{'eval_loss': 0.30023109912872314, 'eval_accuracy_score': 0.897184934544395, 'eval_precision': 0.7095219104439517, 'eval_recall': 0.7496300541014276, 'eval_f1': 0.7290247500409769, 'eval_runtime': 112.8751, 'eval_samples_per_second': 122.268, 'eval_steps_per_second': 1.914, 'epoch': 0.13}
{'loss': 0.2831, 'learning_rate': 9.606815203145478e-05, 'epoch': 0.2}
{'loss': 0.2694, 'learning_rate': 9.475753604193972e-05, 'epoch': 0.26}

