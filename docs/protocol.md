## Preparing training of Soda Roberta bio-lm model using PoS tagginf instead of MLM

```bash
    python -m smtag.cli.prepro.extract /data/xml/oapmc /data/xml/oapmc_articles --xpath .//article --keep_xml
```

I copied the files from the  folder `/raid/lemberge/soda-roberta/data/json/oapmc_abstracts_figs`

The following command is sent to train the language model using the task to find out the small words. 11.06.2022 at 13.36. Training in 4 GPU.

```bash
    python -m smtag.cli.lm.train smtag/loader/loader_lm.py SMALL \
    --data_dir /data/json/oapmc_abstracts_figs --output_dir /lm_models/lm_models_small \
    --overwrite_output_dir --evaluation_strategy steps --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 --learning_rate 0.00005 --num_train_epochs 3 \
    --logging_dir "/runs/lm-SMALL-11-06-2022-13.30" --logging_steps 500 \
    --save_steps 5000 --save_total_limit 5 --eval_steps 5000 --run_name '/lm_models/lm_models_small'
```

The training argumentds used for this training are summarized below:

```config
{ "output_dir": "/lm_models/lm_models_small", "overwrite_output_dir": true, "do_train": false, "do_eval": true, "do_predict": false, "evaluation_strategy": "steps", "prediction_loss_only": true, "per_device_train_batch_size": 8, "per_device_eval_batch_size": 16, "per_gpu_train_batch_size": null, "per_gpu_eval_batch_size": null, "gradient_accumulation_steps": 1, "eval_accumulation_steps": null, "learning_rate": 5e-05, "weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-08, "max_grad_norm": 1.0, "num_train_epochs": 3, "max_steps": -1, "lr_scheduler_type": "linear", "warmup_ratio": 0.0, "warmup_steps": 0, "log_level": -1, "log_level_replica": -1 , "log_on_each_node": true, "logging_dir": "/runs/lm-SMALL-2022-07-12T10-40-30.039887", "logging_strategy": "steps", "logging_first_step": false, "logging_steps": 500, "logging_nan_inf_filter": true, "save_strategy": "steps", "save_steps": 5000, "save_total_limit": 5, "save_on_each_node": false, "no_cuda": false, "seed": 42, "bf16": false, "fp16": false, "fp16_opt_level": "O1", "half_precision_backend": "auto", "bf16_full_eval": false, "fp16_full_eval": false, "tf32": null, "local_rank": -1, "xpu_backend": null, "tpu_num_cores": null, "tpu_metrics_debug": false, "debug": [], "dataloader_drop_last": false, "eval_steps": 5000, "dataloader_num_workers": 0, "past_index": -1, "run_name": "/lm_models/lm_models_small", "disable_tqdm": false, "remove_unused_columns": false, "label_names": null, "load_best_model_at_end": false, "metric_for_best_model": null, "greater_is_better": null, "ignore_data_skip": false, "sharded_ddp": [], "deepspeed": null, "label_smoothing_factor": 0.0, "adafactor": false, "group_by_length": false, "length_column_name": "length", "report_to": [ "tensorboard" ], "ddp_find_unused_parameters": null, "ddp_bucket_cap_mb": null, "dataloader_pin_memory": true, "skip_memory_metrics": true, "use_legacy_prediction_loop": false, "push_to_hub": false, "resume_from_checkpoint": null, "hub_model_id": null, "hub_strategy": "every_save", "hub_token": "<hub_token>", "gradient_checkpointing": false, "fp16_backend": "auto", "push_to_hub_model_id": null, "push_to_hub_organization": null, "push_to_hub_token": "<push_to_hub_token>", "_n_gpu": 4, "mp_parameters": "" }</push_to_hub_token></hub_token>

{ "_name_or_path": "roberta-base", "architectures": [ "RobertaForMaskedLM" ], "attention_probs_dropout_prob": 0.1, "bos_token_id": 0, "classifier_dropout": null, "eos_token_id": 2, "hidden_act": "gelu", "hidden_dropout_prob": 0.1, "hidden_size": 768, "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-05, "max_position_embeddings": 514, "model_type": "roberta", "num_attention_heads": 12, "num_hidden_layers": 12, "pad_token_id": 1, "position_embedding_type": "absolute", "transformers_version": "4.15.0", "type_vocab_size": 1, "use_cache": true, "vocab_size": 50265 }
```

The model has been trained. Now we will check the model in the EMBO dataset and compare it with others, including the MLM version of the language model. 

In the following, the training base parameters will be as following. Any changes will appear in the CLI command.

```bash
TrainingArgumentsTOKCL(
_n_gpu=4,
adafactor=True,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-10,
bf16=False,
bf16_full_eval=False,
classifier_dropout=0.25,
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
do_train=True,
eval_accumulation_steps=None,
eval_steps=500,
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
hub_model_id=None,
hub_strategy=HubStrategy.EVERY_SAVE,
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
logging_dir=/tokcl_models/runs/Jul20_09-03-50_6f1bb9a73a20,
logging_first_step=True,
logging_nan_inf_filter=True,
logging_steps=100,
logging_strategy=IntervalStrategy.STEPS,
lr_schedule=cosine,
lr_scheduler_type=SchedulerType.LINEAR,
masking_probability=None,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=f1,
mp_parameters=,
no_cuda=False,
num_train_epochs=3.0,
output_dir=/tokcl_models,
overwrite_output_dir=True,
past_index=-1,
per_device_eval_batch_size=64,
per_device_train_batch_size=16,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
remove_unused_columns=True,
replacement_probability=None,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=sd-lm-SMALL-NER,
save_on_each_node=False,
save_steps=1000,
save_strategy=IntervalStrategy.STEPS,
save_total_limit=None,
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

```

## Runs of the EMBO/bio-lm 

### NER

Then we sent a training with the previous MLM version of our source data model.

```bash
    python -m smtag.cli.tokcl.train \                                 
        --loader_path "EMBO/sd-nlp" \                                                   
        --task NER \                                                                     
        --from_pretrained "EMBO/bio-lm" \                                                
        --disable_tqdm False \                                                          
        --do_train \                                                                    
        --do_eval \                                                                     
        --do_predict           

                precision    recall  f1-score   support

          CELL       0.72      0.81      0.76      5245
     EXP_ASSAY       0.60      0.63      0.62     10067
      GENEPROD       0.79      0.88      0.83     23587
      ORGANISM       0.74      0.83      0.78      3623
SMALL_MOLECULE       0.73      0.79      0.76      6187
   SUBCELLULAR       0.66      0.74      0.70      3700
        TISSUE       0.65      0.73      0.69      3207

     micro avg       0.72      0.80      0.76     55616
     macro avg       0.70      0.77      0.73     55616
  weighted avg       0.72      0.80      0.76     55616

{'test_loss': 0.2054957002401352, 'test_accuracy_score': 0.9360231178559216, 'test_precision': 0.7228917627489033, 'test_recall': 0.7970188434982739, 'test_f1': 0.7581476888869885, 'test_runtime': 51.653, 'test_s
amples_per_second': 138.966, 'test_steps_per_second': 0.561}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --add_prefix_space \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support

          CELL       0.67      0.78      0.72      4948
     EXP_ASSAY       0.57      0.59      0.58      9885
      GENEPROD       0.77      0.86      0.81     21854
      ORGANISM       0.73      0.83      0.77      3463
SMALL_MOLECULE       0.68      0.79      0.73      6430
   SUBCELLULAR       0.70      0.71      0.71      3850
        TISSUE       0.65      0.72      0.69      2974

     micro avg       0.70      0.77      0.74     53404
     macro avg       0.68      0.75      0.72     53404
  weighted avg       0.70      0.77      0.73     53404


python -m smtag.cli.tokcl.train \
   --loader_path "EMBO/sd-nlp" \
   --task NER \
   --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
   --num_train_epochs 2.0 \
   --disable_tqdm False \
   --do_train \
   --do_eval \
   --do_predict

                precision    recall  f1-score   support

          CELL       0.68      0.81      0.74      5245
     EXP_ASSAY       0.58      0.56      0.57     10067
      GENEPROD       0.77      0.86      0.81     23587
      ORGANISM       0.73      0.81      0.77      3623
SMALL_MOLECULE       0.70      0.78      0.74      6187
   SUBCELLULAR       0.65      0.74      0.69      3700
        TISSUE       0.66      0.71      0.68      3207

     micro avg       0.71      0.77      0.74     55616
     macro avg       0.68      0.75      0.72     55616
  weighted avg       0.71      0.77      0.74     55616

{'test_loss': 0.1906978338956833, 'test_accuracy_score': 0.933576712382461, 'test_precision': 0.7082713232132818, 'test_recall': 0.770138089758343, 'test_f1': 0.7379102420535791, 'test_runtime': 51.3342, 'test_sa
mples_per_second': 139.829, 'test_steps_per_second': 0.565}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --add_prefix_space \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

          CELL       0.66      0.79      0.72      4948
     EXP_ASSAY       0.56      0.62      0.59      9885
      GENEPROD       0.76      0.88      0.82     21854
      ORGANISM       0.71      0.86      0.78      3463
SMALL_MOLECULE       0.68      0.79      0.73      6430
   SUBCELLULAR       0.68      0.75      0.71      3850
        TISSUE       0.64      0.76      0.70      2974

     micro avg       0.69      0.80      0.74     53404
     macro avg       0.67      0.78      0.72     53404
  weighted avg       0.69      0.80      0.74     53404

{'test_loss': 0.2022101730108261, 'test_accuracy_score': 0.9307249085399303, 'test_precision': 0.6926381566100371, 'test_recall': 0.7970189498913939, 'test_f1': 0.7411715539458104, 'test_runtime': 56.7967, 'test_
samples_per_second': 144.815, 'test_steps_per_second': 0.581}

```

### GENEPROD ROLES

Using the SMALL version of our sd model and the roberta-tokenized dataset
```bash

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task GENEPROD_ROLES \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.80      0.87      0.83      7835
  MEASURED_VAR       0.81      0.86      0.84      9330

     micro avg       0.81      0.86      0.83     17165
     macro avg       0.80      0.86      0.83     17165
  weighted avg       0.81      0.86      0.83     17165

{'test_loss': 0.04704853147268295, 'test_accuracy_score': 0.98534319284298, 'test_precision': 0.8054045237059592, 'test_recall': 0.8629769880570929, 'test_f1': 0.8331974013555699, 'test_runtime': 49.6416, 'test_s
amples_per_second': 144.596, 'test_steps_per_second': 0.584}
```

Same as above but using the word pre-tokenized dataset

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --add_prefix_space \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support

CONTROLLED_VAR       0.90      0.90      0.90      7238
  MEASURED_VAR       0.91      0.93      0.92      8718

     micro avg       0.90      0.92      0.91     15956
     macro avg       0.90      0.92      0.91     15956
  weighted avg       0.90      0.92      0.91     15956

{'test_loss': 0.013265026733279228, 'test_accuracy_score': 0.9950212429489678, 'test_precision': 0.9042967542503864, 'test_recall': 0.9167084482326397, 'test_f1': 0.9104603031340449, 'test_runtime': 56.0586, 'tes
t_samples_per_second': 146.722, 'test_steps_per_second': 0.589}

```

Using the masked language model version and the roberta-tokenized data.

```bash

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task GENEPROD_ROLES \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.80      0.88      0.84      7835
  MEASURED_VAR       0.83      0.85      0.84      9330

     micro avg       0.81      0.86      0.84     17165
     macro avg       0.81      0.86      0.84     17165
  weighted avg       0.82      0.86      0.84     17165

{'test_loss': 0.04671486094594002, 'test_accuracy_score': 0.9855999978374316, 'test_precision': 0.8149023920813857, 'test_recall': 0.8633265365569472, 'test_f1': 0.8384158415841584, 'test_runtime': 49.879, 'test_
samples_per_second': 143.908, 'test_steps_per_second': 0.581}
```

Using the masked language model version and word pre-tokenized data adding space to the words.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --add_prefix_space \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.91      0.92      0.91      7238
  MEASURED_VAR       0.93      0.93      0.93      8718

     micro avg       0.92      0.92      0.92     15956
     macro avg       0.92      0.92      0.92     15956
  weighted avg       0.92      0.92      0.92     15956

{'test_loss': 0.01625009812414646, 'test_accuracy_score': 0.9952772428682584, 'test_precision': 0.9204467461159294, 'test_recall': 0.9245424918525946, 'test_f1': 0.9224900728512022, 'test_runtime': 56.4166, 'test
_samples_per_second': 145.79, 'test_steps_per_second': 0.585}

```

### SMALL MOL ROLES

Using the MLM version of our sd model and the word pre-tokenized dataset.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "EMBO/bio-lm" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.91      0.92      0.91       893

     micro avg       0.96      0.97      0.96      4187
     macro avg       0.94      0.95      0.95      4187
  weighted avg       0.96      0.97      0.96      4187

{'test_loss': 0.0020785119850188494, 'test_accuracy_score': 0.9995321380785378, 'test_precision': 0.9626457292410183, 'test_recall': 0.9663243372342967, 'test_f1': 0.9644815256257449, 'test_runtime': 55.7017, 'te
st_samples_per_second': 147.662, 'test_steps_per_second': 0.592}

```

Using the MLM version of our sd model and the roberta-tokenized dataset.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.82      0.90      0.86      2946
  MEASURED_VAR       0.73      0.81      0.77       852

     micro avg       0.80      0.88      0.84      3798
     macro avg       0.77      0.85      0.81      3798
  weighted avg       0.80      0.88      0.84      3798

{'test_loss': 0.013387518934905529, 'test_accuracy_score': 0.995937074666728, 'test_precision': 0.7983677388382141, 'test_recall': 0.875724065297525, 'test_f1': 0.8352586639879458, 'test_runtime': 48.9263, 'test_
samples_per_second': 146.711, 'test_steps_per_second': 0.593}
```

SD bio language trained with SMALL and the roberta-tokenized data.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.80      0.90      0.85      2946
  MEASURED_VAR       0.70      0.77      0.73       852

     micro avg       0.78      0.87      0.82      3798
     macro avg       0.75      0.83      0.79      3798
  weighted avg       0.78      0.87      0.82      3798

{'test_loss': 0.01325817871838808, 'test_accuracy_score': 0.9957492015392081, 'test_precision': 0.7768381489311722, 'test_recall': 0.8707214323328067, 'test_f1': 0.821104903786468, 'test_runtime': 49.1015, 'test_
samples_per_second': 146.187, 'test_steps_per_second': 0.591}
```

SD bio language trained with SMALL and the word pre-tokenized data. 
```bash
python -m smtag.cli.tokcl.train \                                                                                                                                                           
>     --loader_path "EMBO/sd-nlp-non-tokenized" \                                                                                                                                                                   
>     --task SMALL_MOL_ROLES \                                                                                                                                                                                      
>     --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \                                                                                                                                           
>     --add_prefix_space \                                                                                                                                                                                          
>     --num_train_epochs 2.0 \                                                                                                                                                                                      
>     --disable_tqdm False \                                                                                                                                                                                        
>     --masked_data_collator \                                                                                                                                                                                      
>     --do_train \                                                                                                                                                                                                  
>     --do_eval \                                                                                                                                                                                                   
>     --do_predict           

                precision    recall  f1-score   support

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.91      0.92      0.91       893

     micro avg       0.96      0.96      0.96      4187
     macro avg       0.94      0.95      0.95      4187
  weighted avg       0.96      0.96      0.96      4187

{'test_loss': 0.0022225005086511374, 'test_accuracy_score': 0.9995018720782275, 'test_precision': 0.9619047619047619, 'test_recall': 0.9648913303080965, 'test_f1': 0.9633957314892094, 'test_runtime': 55.6241, 'te
st_samples_per_second': 147.867, 'test_steps_per_second': 0.593}

```

### PANELIZATION TASK IN SD bio language

SMALL version and roberta-tokenized dataset.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task PANELIZATION \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.91      0.94      0.93      5427

   micro avg       0.91      0.94      0.93      5427
   macro avg       0.91      0.94      0.93      5427
weighted avg       0.91      0.94      0.93      5427

{'test_loss': 0.003634429769590497, 'test_accuracy_score': 0.9987811641009996, 'test_precision': 0.914065303193398, 'test_recall': 0.9388243965358393, 'test_f1': 0.9262794291428051, 'test_runtime': 31.3935, 'test
_samples_per_second': 57.4, 'test_steps_per_second': 0.255}

```

MLM version and roberta-tokenized dataset

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task PANELIZATION \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.91      0.95      0.93      5427

   micro avg       0.91      0.95      0.93      5427
   macro avg       0.91      0.95      0.93      5427
weighted avg       0.91      0.95      0.93      5427

{'test_loss': 0.0033620719332247972, 'test_accuracy_score': 0.9988412793118011, 'test_precision': 0.9098591549295775, 'test_recall': 0.9522756587433204, 'test_f1': 0.9305843161969929, 'test_runtime': 31.8429, 'te
st_samples_per_second': 56.59, 'test_steps_per_second': 0.251}

```

SMALL version and word pre-tokenized dataset.

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --add_prefix_space \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.92      0.94      0.93      7724

   micro avg       0.92      0.94      0.93      7724
   macro avg       0.92      0.94      0.93      7724
weighted avg       0.92      0.94      0.93      7724

{'test_loss': 0.005392521619796753, 'test_accuracy_score': 0.9981677394004365, 'test_precision': 0.9211928934010152, 'test_recall': 0.9397980321077162, 'test_f1': 0.9304024609074596, 'test_runtime': 33.3522, 'tes
t_samples_per_second': 57.507, 'test_steps_per_second': 0.24}

```

MLM version and word pre-tokenized dataset

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm False \
    --add_prefix_space \
    --do_train \
    --do_eval \
    --do_predict 
    
               precision    recall  f1-score   support

 PANEL_START       0.93      0.94      0.93      7724

   micro avg       0.93      0.94      0.93      7724
   macro avg       0.93      0.94      0.93      7724
weighted avg       0.93      0.94      0.93      7724

{'test_loss': 0.005239087622612715, 'test_accuracy_score': 0.9982702434899225, 'test_precision': 0.9286445012787724, 'test_recall': 0.9401864319005696, 'test_f1': 0.9343798250128668, 'test_runtime': 33.4133, 'tes
t_samples_per_second': 57.402, 'test_steps_per_second': 0.239}
   
```




In every model from now on we will be using only the word  pre-tokenized dataset `EMBO/sd-nlp-non-tokenized`.

## PubMedBERT (abstract and full text)

### GENEPROD ROLES

```bash

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.92      0.92      0.92      7241
  MEASURED_VAR       0.93      0.93      0.93      8720

     micro avg       0.92      0.93      0.93     15961
     macro avg       0.92      0.93      0.93     15961
  weighted avg       0.92      0.93      0.93     15961
  {'test_loss': 0.010478646494448185, 'test_accuracy_score': 0.9963540958289276, 'test_precision': 0.9242244554022845, 'test_recall': 0.9276987657414949, 'test_f1': 0.9259583515727596, 'test_runtime': 51.3749, 'tes
t_samples_per_second': 160.098, 'test_steps_per_second': 0.642}
```

### PANELIZATION

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 
              precision    recall  f1-score   support

 PANEL_START       0.92      0.95      0.93      7990

   micro avg       0.92      0.95      0.93      7990
   macro avg       0.92      0.95      0.93      7990
weighted avg       0.92      0.95      0.93      7990

{'test_loss': 0.0047940718941390514, 'test_accuracy_score': 0.9982366494772645, 'test_precision': 0.9177238129757158, 'test_recall': 0.9506883604505632, 'test_f1': 0.9339152886211348, 'test_runtime': 31.5407, 'te
st_samples_per_second': 60.81, 'test_steps_per_second': 0.254}
```

### SMALL_MOL_ROLES

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support

CONTROLLED_VAR       0.97      0.98      0.98      3294
  MEASURED_VAR       0.93      0.90      0.92       894

     micro avg       0.96      0.97      0.96      4188
     macro avg       0.95      0.94      0.95      4188
  weighted avg       0.96      0.97      0.96      4188

{'test_loss': 0.0016298561822623014, 'test_accuracy_score': 0.9996475479722309, 'test_precision': 0.9642175572519084, 'test_recall': 0.9651384909264565, 'test_f1': 0.9646778042959427, 'test_runtime': 50.6603, 'te
st_samples_per_second': 162.356, 'test_steps_per_second': 0.651}
```

## BioBERT base and large

### GENEPROD_ROLES

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-base" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                    precision    recall  f1-score   support

CONTROLLED_VAR       0.90      0.93      0.92      7241
  MEASURED_VAR       0.94      0.92      0.93      8720

     micro avg       0.92      0.93      0.92     15961
     macro avg       0.92      0.93      0.92     15961
  weighted avg       0.92      0.93      0.92     15961

{'test_loss': 0.011445160955190659, 'test_accuracy_score': 0.995920591294329, 'test_precision': 0.9205785175487813, 'test_recall': 0.9251926571016854, 'test_f1': 0.9228798200112494, 'test_runtime': 52.4485, 'test
_samples_per_second': 156.82, 'test_steps_per_second': 0.629}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.00005 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 
 
                precision    recall  f1-score   support

CONTROLLED_VAR       0.91      0.92      0.92      7241
  MEASURED_VAR       0.93      0.93      0.93      8720

     micro avg       0.92      0.93      0.92     15961
     macro avg       0.92      0.93      0.92     15961
  weighted avg       0.92      0.93      0.92     15961
```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "michiyasunaga/BioLinkBERT-base" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.91      0.95      0.93      7948

   micro avg       0.91      0.95      0.93      7948
   macro avg       0.91      0.95      0.93      7948
weighted avg       0.91      0.95      0.93      7948

{'test_loss': 0.005119128618389368, 'test_accuracy_score': 0.9981422798678409, 'test_precision': 0.9121744397334949, 'test_recall': 0.947408152994464, 'test_f1': 0.9294575078689132, 'test_runtime': 32.0476, 'test
_samples_per_second': 59.849, 'test_steps_per_second': 0.25}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --learning_rate 0.00005 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.93      0.95      0.94      7948

   micro avg       0.93      0.95      0.94      7948
   macro avg       0.93      0.95      0.94      7948
weighted avg       0.93      0.95      0.94      7948

{'test_loss': 0.004488417878746986, 'test_accuracy_score': 0.998386393505042, 'test_precision': 0.9288357178095707, 'test_recall': 0.9475339708102667, 'test_f1': 0.9380916791230691, 'test_runtime': 42.2304, 'test
_samples_per_second': 45.417, 'test_steps_per_second': 0.189}
```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.92      0.93      0.92       894

     micro avg       0.97      0.97      0.97      4188
     macro avg       0.95      0.95      0.95      4188
  weighted avg       0.97      0.97      0.97      4188

{'test_loss': 0.001487505272962153, 'test_accuracy_score': 0.9996381741843666, 'test_precision': 0.965673420738975, 'test_recall': 0.9672874880611271, 'test_f1': 0.9664797805081714, 'test_runtime': 51.8547, 'test
_samples_per_second': 158.616, 'test_steps_per_second': 0.636}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --masked_data_collator \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.92      0.92      0.92       894

     micro avg       0.96      0.97      0.97      4188
     macro avg       0.95      0.95      0.95      4188
  weighted avg       0.97      0.97      0.97      4188

{'test_loss': 0.0012178965844213963, 'test_accuracy_score': 0.9996466877329697, 'test_precision': 0.9649666348903718, 'test_recall': 0.9668099331423113, 'test_f1': 0.9658874045801527
, 'test_runtime': 52.6042, 'test_samples_per_second': 156.356, 'test_steps_per_second': 0.627}

```

## BioBERT (Large and base)

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "dmis-lab/biobert-large-cased-v1.1" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.00005 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.91      0.92      0.91      7238
  MEASURED_VAR       0.93      0.93      0.93      8718

     micro avg       0.92      0.92      0.92     15956
     macro avg       0.92      0.92      0.92     15956
  weighted avg       0.92      0.92      0.92     15956

{'test_loss': 0.013223936781287193, 'test_accuracy_score': 0.9955241753588387, 'test_precision': 0.9182491750202354, 'test_recall': 0.9242918024567561, 'test_f1': 0.9212605803167068, 'test_runtime': 93.2093, 'tes
t_samples_per_second': 88.242, 'test_steps_per_second': 0.354}
```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "dmis-lab/biobert-large-cased-v1.1" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --learning_rate 0.000025 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 
              precision    recall  f1-score   support

 PANEL_START       0.92      0.95      0.93      7699

   micro avg       0.92      0.95      0.93      7699
   macro avg       0.92      0.95      0.93      7699
weighted avg       0.92      0.95      0.93      7699

{'test_loss': 0.005264635197818279, 'test_accuracy_score': 0.9981635715759003, 'test_precision': 0.916185696361355, 'test_recall': 0.9484348616703469, 'test_f1': 0.9320313995787862, 'test_runtime': 44.363, 'test_
samples_per_second': 43.234, 'test_steps_per_second': 0.18}
```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "dmis-lab/biobert-large-cased-v1.1" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.000025 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 


                precision    recall  f1-score   support

CONTROLLED_VAR       0.97      0.98      0.98      3293
  MEASURED_VAR       0.92      0.88      0.90       893

     micro avg       0.96      0.96      0.96      4186
     macro avg       0.94      0.93      0.94      4186
  weighted avg       0.96      0.96      0.96      4186

{'test_loss': 0.0021991219837218523, 'test_accuracy_score': 0.9995514122973345, 'test_precision': 0.9569969113803753, 'test_recall': 0.9622551361681796, 'test_f1': 0.9596188207266229, 'test_runtime': 93.338, 'tes
t_samples_per_second': 88.121, 'test_steps_per_second': 0.354}

```


## RoBERTa base

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "roberta-base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 


                precision    recall  f1-score   support

CONTROLLED_VAR       0.89      0.90      0.90      7238
  MEASURED_VAR       0.92      0.92      0.92      8718

     micro avg       0.91      0.91      0.91     15956
     macro avg       0.91      0.91      0.91     15956
  weighted avg       0.91      0.91      0.91     15956

{'test_loss': 0.01501127053052187, 'test_accuracy_score': 0.9945634696942756, 'test_precision': 0.9060632470119522, 'test_recall': 0.9121960391075458, 'test_f1': 0.9091193004372268, 'test_runtime': 56.8377, 'test
_samples_per_second': 144.71, 'test_steps_per_second': 0.581}
```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "roberta-base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.90      0.95      0.93      7724

   micro avg       0.90      0.95      0.93      7724
   macro avg       0.90      0.95      0.93      7724
weighted avg       0.90      0.95      0.93      7724

{'test_loss': 0.005761752370744944, 'test_accuracy_score': 0.9979371051990928, 'test_precision': 0.9023523646165156, 'test_recall': 0.9535214914552046, 'test_f1': 0.9272315246128667, 'test_runtime': 34.3078, 'tes
t_samples_per_second': 55.906, 'test_steps_per_second': 0.233} 
```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "roberta-base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.97      0.98      0.98      3294
  MEASURED_VAR       0.92      0.89      0.90       893

     micro avg       0.96      0.96      0.96      4187
     macro avg       0.94      0.94      0.94      4187
  weighted avg       0.96      0.96      0.96      4187

{'test_loss': 0.0021583831403404474, 'test_accuracy_score': 0.9994463844109921, 'test_precision': 0.9585615622767325, 'test_recall': 0.9613088129925962, 'test_f1': 0.9599332220367278, 'test_runtime': 55.7043, 'te
st_samples_per_second': 147.655, 'test_steps_per_second': 0.592}
```

## BERT base cased 

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "bert-base-cased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

CONTROLLED_VAR       0.89      0.89      0.89      7233
  MEASURED_VAR       0.90      0.91      0.91      8716

     micro avg       0.89      0.90      0.90     15949
     macro avg       0.89      0.90      0.90     15949
  weighted avg       0.89      0.90      0.90     15949

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "bert-base-cased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "bert-base-cased_PANELIZATION"

              precision    recall  f1-score   support

 PANEL_START       0.86      0.87      0.87      7497

   micro avg       0.86      0.87      0.87      7497
   macro avg       0.86      0.87      0.87      7497
weighted avg       0.86      0.87      0.87      7497

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "bert-base-cased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "bert-base-cased_SMALL_MOL_ROLES"

                precision    recall  f1-score   support

CONTROLLED_VAR       0.96      0.98      0.97      3292
  MEASURED_VAR       0.90      0.85      0.87       893

     micro avg       0.94      0.95      0.95      4185
     macro avg       0.93      0.91      0.92      4185
  weighted avg       0.94      0.95      0.95      4185

```

## BERT base uncased 

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "bert-base-uncased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "bert-base-uncased_GENEPROD_ROLES"

                    precision    recall  f1-score   support

CONTROLLED_VAR       0.89      0.90      0.90      7237
  MEASURED_VAR       0.91      0.92      0.92      8718

     micro avg       0.90      0.91      0.91     15955
     macro avg       0.90      0.91      0.91     15955
  weighted avg       0.90      0.91      0.91     15955

{'test_loss': 0.01473652757704258, 'test_accuracy_score': 0.9946272341596362, 'test_precision': 0.9044566164571144, 'test_recall': 0.9107489815104983, 'test_f1': 0.9075918928203367, 'te
st_runtime': 58.4083, 'test_samples_per_second': 140.819, 'test_steps_per_second': 0.565}

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "bert-base-uncased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "bert-base-uncased_PANELIZATION"
              precision    recall  f1-score   support

 PANEL_START       0.89      0.93      0.91      7662

   micro avg       0.89      0.93      0.91      7662
   macro avg       0.89      0.93      0.91      7662
weighted avg       0.89      0.93      0.91      7662

{'test_loss': 0.0073542520403862, 'test_accuracy_score': 0.9973646825133705, 'test_precision': 0.8870465029298092, 'test_recall': 0.9286087183503002, 'test_f1': 0.9073519097111523, 'tes
t_runtime': 34.3427, 'test_samples_per_second': 55.849, 'test_steps_per_second': 0.233}

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "bert-base-uncased" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "bert-base-uncased_SMALL_MOL_ROLES"

                precision    recall  f1-score   support

CONTROLLED_VAR       0.97      0.97      0.97      3292
  MEASURED_VAR       0.90      0.88      0.89       893

     micro avg       0.95      0.95      0.95      4185
     macro avg       0.93      0.93      0.93      4185
  weighted avg       0.95      0.95      0.95      4185

{'test_loss': 0.002427967032417655, 'test_accuracy_score': 0.9993344463076643, 'test_precision': 0.9507260176148536, 'test_recall': 0.9543608124253286, 'test_f1': 0.9525399475316003, 't
est_runtime': 57.9758, 'test_samples_per_second': 141.87, 'test_steps_per_second': 0.569}

```

## BioBERT base cased 

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "dmis-lab/biobert-base-cased-v1.2" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "biobert-base-cased_GENEPROD_ROLES"
                precision    recall  f1-score   support

CONTROLLED_VAR       0.89      0.90      0.90      7231
  MEASURED_VAR       0.91      0.92      0.92      8715

     micro avg       0.90      0.91      0.91     15946
     macro avg       0.90      0.91      0.91     15946
  weighted avg       0.90      0.91      0.91     15946

{'test_loss': 0.016174716874957085, 'test_accuracy_score': 0.994354416656227, 'test_precision': 0.9041394335511983, 'test_recall': 0.9108867427568043, 'test_f1': 0.9075005466870764, 'te
st_runtime': 60.9812, 'test_samples_per_second': 134.878, 'test_steps_per_second': 0.541}

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "dmis-lab/biobert-base-cased-v1.2" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "biobert-base-cased_PANELIZATION"
              precision    recall  f1-score   support

 PANEL_START       0.90      0.94      0.92      7524

   micro avg       0.90      0.94      0.92      7524
   macro avg       0.90      0.94      0.92      7524
weighted avg       0.90      0.94      0.92      7524

{'test_loss': 0.006445760373026133, 'test_accuracy_score': 0.9977361996297587, 'test_precision': 0.8950094756790903, 'test_recall': 0.9415204678362573, 'test_f1': 0.917676015285964, 'te
st_runtime': 35.1944, 'test_samples_per_second': 54.497, 'test_steps_per_second': 0.227}

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "dmis-lab/biobert-base-cased-v1.2" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "biobert-base-cased_SMALL_MOL_ROLES"
                precision    recall  f1-score   support

CONTROLLED_VAR       0.97      0.98      0.98      3291
  MEASURED_VAR       0.92      0.90      0.91       893

     micro avg       0.96      0.96      0.96      4184
     macro avg       0.94      0.94      0.94      4184
  weighted avg       0.96      0.96      0.96      4184

{'test_loss': 0.002317107282578945, 'test_accuracy_score': 0.9994356758250315, 'test_precision': 0.9586403613025909, 'test_recall': 0.9639101338432122, 'test_f1': 0.961268025265165, 
'test_runtime': 60.1791, 'test_samples_per_second': 136.675, 'test_steps_per_second': 0.548}
    

```
## PubMedBERT base abstracts

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "PMB-abstract_GENEPROD_ROLES"

                precision    recall  f1-score   support

CONTROLLED_VAR       0.91      0.93      0.92      7241
  MEASURED_VAR       0.94      0.93      0.93      8720

     micro avg       0.93      0.93      0.93     15961
     macro avg       0.93      0.93      0.93     15961
  weighted avg       0.93      0.93      0.93     15961

{'test_loss': 0.010013540275394917, 'test_accuracy_score': 0.9965061913097907, 'test_precision': 0.9263980006248047, 'test_recall': 0.9289518200613996, 'test_f1': 0.9276731527247701,
 'test_runtime': 51.6423, 'test_samples_per_second': 159.269, 'test_steps_per_second': 0.639}

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "PMB-abstract_PANELIZATION"

              precision    recall  f1-score   support

 PANEL_START       0.92      0.95      0.94      7990

   micro avg       0.92      0.95      0.94      7990
   macro avg       0.92      0.95      0.94      7990
weighted avg       0.92      0.95      0.94      7990

{'test_loss': 0.0046697696670889854, 'test_accuracy_score': 0.9983040486528002, 'test_precision': 0.922292374939291, 'test_recall': 0.9506883604505632, 'test_f1': 0.9362751140145446,
 'test_runtime': 31.5726, 'test_samples_per_second': 60.749, 'test_steps_per_second': 0.253}

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "PMB-abstract_SMALL_MOL_ROLES"
    
                precision    recall  f1-score   support

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.93      0.93      0.93       894

     micro avg       0.97      0.97      0.97      4188
     macro avg       0.96      0.95      0.96      4188
  weighted avg       0.97      0.97      0.97      4188

{'test_loss': 0.0013107596896588802, 'test_accuracy_score': 0.9997016588644609, 'test_precision': 0.9692270992366412, 'test_recall': 0.970152817574021, 'test_f1': 0.969689737470167, 'test_runtime': 50.9966, 'test_samples_per_second': 161.285, 'test_steps_per_second':
 0.647}

```

## BioMegatron

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "EMBO/BioMegatron345mUncased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.000025 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mUncased_GENEPROD_ROLES"

                precision    recall  f1-score   support

CONTROLLED_VAR       0.90      0.91      0.91      7237
  MEASURED_VAR       0.92      0.93      0.92      8718

     micro avg       0.91      0.92      0.92     15955
     macro avg       0.91      0.92      0.92     15955
  weighted avg       0.91      0.92      0.92     15955

{'test_loss': 0.013211735524237156, 'test_accuracy_score': 0.995750123629387, 'test_precision': 0.9079660182221128, 'test_recall': 0.9244124099028518, 'test_f1': 0.916115407310786, '
test_runtime': 101.4736, 'test_samples_per_second': 81.056, 'test_steps_per_second': 0.325}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "EMBO/BioMegatron345mCased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.000025 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mCased_GENEPROD_ROLES"

                precision    recall  f1-score   support

CONTROLLED_VAR       0.90      0.93      0.91      7233
  MEASURED_VAR       0.93      0.93      0.93      8716

     micro avg       0.91      0.93      0.92     15949
     macro avg       0.91      0.93      0.92     15949
  weighted avg       0.92      0.93      0.92     15949

{'test_loss': 0.012827829457819462, 'test_accuracy_score': 0.9956606382274786, 'test_precision': 0.9148436534420962, 'test_recall': 0.9282086651200703, 'test_f1': 0.9214777006629111,
 'test_runtime': 106.6473, 'test_samples_per_second': 77.123, 'test_steps_per_second': 0.309}

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "EMBO/BioMegatron345mCased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --learning_rate 0.000025 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mCased_PANELIZATION"

              precision    recall  f1-score   support

 PANEL_START       0.92      0.94      0.93      7497

   micro avg       0.92      0.94      0.93      7497
   macro avg       0.92      0.94      0.93      7497
weighted avg       0.92      0.94      0.93      7497

{'test_loss': 0.005305187776684761, 'test_accuracy_score': 0.9982626238458275, 'test_precision': 0.9200883575883576, 'test_recall': 0.9445111377884488, 'test_f1': 0.932139801224248, 
'test_runtime': 47.7397, 'test_samples_per_second': 40.176, 'test_steps_per_second': 0.168}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "EMBO/BioMegatron345mUncased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --learning_rate 0.000025 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mUncased_PANELIZATION"

              precision    recall  f1-score   support

 PANEL_START       0.92      0.95      0.93      7662

   micro avg       0.92      0.95      0.93      7662
   macro avg       0.92      0.95      0.93      7662
weighted avg       0.92      0.95      0.93      7662

{'test_loss': 0.005312655121088028, 'test_accuracy_score': 0.9982788611704027, 'test_precision': 0.9185589242674109, 'test_recall': 0.9450535108326807, 'test_f1': 0.9316178835638469,
 'test_runtime': 46.814, 'test_samples_per_second': 40.971, 'test_steps_per_second': 0.171}

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "EMBO/BioMegatron345mCased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.000025 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mCased_SMALL_MOL_ROLES"

                    precision    recall  f1-score   support                                                                                                                                                             
                                                                                                                                                                                                                    
CONTROLLED_VAR       0.97      0.98      0.98      3292                                                                                                                                                             
  MEASURED_VAR       0.93      0.91      0.92       893

     micro avg       0.96      0.97      0.96      4185
     macro avg       0.95      0.94      0.95      4185
  weighted avg       0.96      0.97      0.96      4185

{'test_loss': 0.0023707891814410686, 'test_accuracy_score': 0.9995577055015167, 'test_precision': 0.9626368396001904, 'test_recall': 0.966547192353644, 'test_f1': 0.9645880529390723, 'test_runtime': 106.1585, 'te
st_samples_per_second': 77.479, 'test_steps_per_second': 0.311}

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "EMBO/BioMegatron345mUncased" \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.000025 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "BioMegatron345mUnased_SMALL_MOL_ROLES"

                precision    recall  f1-score   support                                                                                                                                                             
                                                                                                                                                                                                                    
CONTROLLED_VAR       0.97      0.98      0.98      3292                                                                                                                                                             
  MEASURED_VAR       0.93      0.90      0.91       893                                                                                                                                                             
                                                                                                                                                                                                                    
     micro avg       0.96      0.96      0.96      4185
     macro avg       0.95      0.94      0.94      4185
  weighted avg       0.96      0.96      0.96      4185

{'test_loss': 0.0020676676649600267, 'test_accuracy_score': 0.9995624945001964, 'test_precision': 0.9602759276879163, 'test_recall': 0.9646356033452808, 'test_f1': 0.9624508284658481, 'test_runtime': 101.3689, 't
est_samples_per_second': 81.139, 'test_steps_per_second': 0.326}

```

## Runs of the biomed-roberta-base model

### NER

```bsh
   python -m smtag.cli.tokcl.train \
      --loader_path "EMBO/sd-nlp-non-tokenized" \
      --task NER \
      --from_pretrained "allenai/biomed_roberta_base" \
      --disable_tqdm True \
      --hyperparameter_search \
      --add_prefix_space \
      --hp_experiment_name "EMBO_bio-lm_NER" \
      --hp_gpus_per_trial 1 \
      --hp_tune_samples 16 


python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "allenai/biomed_roberta_base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

```

### GENEPROD ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "allenai/biomed_roberta_base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 
                precision    recall  f1-score   support                                                                                                                                                                             

CONTROLLED_VAR       0.90      0.91      0.90      7238
  MEASURED_VAR       0.92      0.92      0.92      8718

     micro avg       0.91      0.92      0.91     15956
     macro avg       0.91      0.92      0.91     15956
  weighted avg       0.91      0.92      0.91     15956

{'test_loss': 0.014319841749966145, 'test_accuracy_score': 0.9948686518640704, 'test_precision': 0.9091983332296785, 'test_recall': 0.9162070694409626, 'test_f1': 0.9126892461370375, 'test_runtime': 56.8217, 'test_samples_per_se
cond': 144.751, 'test_steps_per_second': 0.581}

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "allenai/biomed_roberta_base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict 

              precision    recall  f1-score   support

 PANEL_START       0.91      0.95      0.93      7724

   micro avg       0.91      0.95      0.93      7724
   macro avg       0.91      0.95      0.93      7724
weighted avg       0.91      0.95      0.93      7724

{'test_loss': 0.005609080661088228, 'test_accuracy_score': 0.9980552696355838, 'test_precision': 0.9092260061919505, 'test_recall': 0.9505437597099948, 'test_f1': 0.9294259130324705, 'test_runtime': 34.1154, 'test_samples_per_se
cond': 56.221, 'test_steps_per_second': 0.234}

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "allenai/biomed_roberta_base" \
    --per_device_train_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support                                                                                                                                                                             

CONTROLLED_VAR       0.98      0.98      0.98      3294
  MEASURED_VAR       0.91      0.91      0.91       893

     micro avg       0.96      0.97      0.96      4187
     macro avg       0.95      0.95      0.95      4187
  weighted avg       0.96      0.97      0.96      4187

{'test_loss': 0.002258655149489641, 'test_accuracy_score': 0.9995081774949588, 'test_precision': 0.9623719933317456, 'test_recall': 0.9651301647957965, 'test_f1': 0.9637491056522774, 'test_runtime': 55.9045, 'test_samples_per_se
cond': 147.126, 'test_steps_per_second': 0.59}

```

### Trainning model to send to production

We now finally train BioLinkBERT to be sent into the EEB and improve the quality of our graph.

### NER

```bsh
python -m smtag.cli.tokcl.train \
    --output_dir "sd-ner-v2" \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --per_device_train_batch_size 8 \
    --save_strategy "epoch" \
    --add_prefix_space \
    --evaluation_strategy "epoch" \
    --num_train_epochs 2.0 \
    --learning_rate 0.00005 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "sd-ner-v2" \
    --push_to_hub \
    --hub_model_id "EMBO/sd-ner-v2" \
    --hub_strategy "end" \
    --hub_token ""

            precision    recall  f1-score support
CELL           0.71      0.79      0.75      4948 
EXP_ASSAY      0.59      0.60      0.60      9885  
GENEPROD       0.79      0.89      0.84     21865                                                                       
ORGANISM       0.72      0.85      0.78      3464  
SMALL_MOLECULE 0.72      0.81      0.76      6431
SUBCELLULAR    0.72      0.77      0.74      3850 
TISSUE         0.68      0.76      0.72      2975 

micro avg       0.72      0.80      0.76
macro avg       0.70      0.78      0.74     53418 
weighted avg    0.72      0.80      0.76     53418 
{'test_loss': 0.16807569563388824, 'test_accuracy_score': 0.9427137503742414, 'test_precision': 0.7242540660382148, 'test_recall': 0.8011157287805608, 'test_f1': 0.7607484111817252, 'test_runtime': 88.1851, 'test_samples_per_second': 93.27, 'test_steps_per_second': 0.374}

### GENEPROD_ROLES

```bash
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task GENEPROD_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --per_device_train_batch_size 8 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.00005 \
    --lr_schedule "cosine" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "sd-geneprod-roles-v2" \
    --push_to_hub \
    --hub_model_id "EMBO/sd-geneprod-roles-v2" \
    --hub_strategy "end" \
    --hub_token ""

                  precision recall    f1-score   support
CONTROLLED_VAR    0.91      0.93      0.92       7241
MEASURED_VAR      0.94      0.93      0.93       872
micro avg         0.92      0.93      0.93       15961
macro avg         0.92      0.93      0.93       15961
weighted avg      0.92      0.93      0.93       15961
{'test_loss': 0.011066839098930359, 'test_accuracy_score': 0.9961050515140637, 'test_precision': 0.9227847313033191, 'test_recall': 0.9284505983334378, 'test_f1': 0.9256089943785134, 'test_runtime': 86.746, 'test_samples_per_second': 94.817, 'test_steps_per_second': 0.38}```
                                                                                                   

```

### PANELIZATION

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --per_device_train_batch_size 8 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --add_prefix_space \
    --learning_rate 0.00005 \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "biolinkBERT_PANELIZATION" \
    --push_to_hub \
    --hub_model_id "EMBO/sd-panelization-v2" \
    --hub_strategy "end" \
    --hub_token ""

```

### SMALL_MOL_ROLES

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task SMALL_MOL_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --add_prefix_space \
    --per_device_train_batch_size 4 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "biolinkBERT_SMALL_MOL_ROLES" \
    --push_to_hub \
    --hub_model_id "EMBO/sd-smallmol-roles-v2" \
    --hub_strategy "end" \
    --hub_token ""

```

# Improving NER performance

NER is currently the bottleneck of our performance. One of the possible issues about it is the inbalance between 
classes. We will act on this following different approaches. First, by adding class weights to the training algorithm.
Then, we will keep the training without class weights, but we will select a balanced version of the dataset. The last
option will be to use ontologies to generate data augmentation. We define below how each of the approaches went down.

We will use PubMedBERT as the model to go in the entire process. The main reason is that it is as good as BioLinkBERT-large
while having only half of its parameters.

## Label smoothing in training parameters

The label smoothing improves the results. The improvement is quite small, but it is enought to generate a 
model that is better than that of BioLinkBERT large.

The label smoothing factor is defined as follows in the documentation.

```
      label_smoothing_factor (`float`, *optional*, defaults to 0.0):
         The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
         labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor +
         label_smoothing_factor/num_labels` respectively.
```

As it can be seen, it has nothing to do with the data balance or inbalance. It is just about the weight of the different types of classes.
This improves the algorithm but that is only part of the story. The next step is to get actually balanced data to see if that improves the options of the 
trainer to generate a better model.

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --label_smoothing_factor 0.25 \
    --per_device_train_batch_size 16 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --learning_rate 0.0001 \
    --lr_schedule "cosine" \
    --disable_tqdm False \
    --run_name "ner-pubmedbert-labelsmoothing" \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

          CELL       0.71      0.81      0.76      4948
     EXP_ASSAY       0.60      0.62      0.61      9885
      GENEPROD       0.79      0.90      0.84     21869
      ORGANISM       0.75      0.87      0.81      3464
SMALL_MOLECULE       0.72      0.82      0.77      6431
   SUBCELLULAR       0.74      0.76      0.75      3850
        TISSUE       0.68      0.76      0.72      2975

     micro avg       0.73      0.81      0.77     53422
     macro avg       0.71      0.79      0.75     53422
  weighted avg       0.73      0.81      0.77     53422

{'test_loss': 1.257552146911621, 'test_accuracy_score': 0.943930878453774, 'test_precision': 0.729350090731904, 'test_recall': 0.8125491370596384, 'test_f1': 0.768704953160141, 'test_runtime': 52.5044, 'test_samp
les_per_second': 156.654, 'test_steps_per_second': 0.629}
```

## Adding class weights to the loss computation in training

Classes might be unbalanced. In these cases, it is important to weight the classes properly. 
For this, we have generated an special trainer that adds class weights to the `CrossEntropyLoss` 
function. These weights are automatically obtained from the Huggingface datasets. 

We check now the performance of this function.

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --per_device_train_batch_size 16 \
    --save_strategy "epoch" \
    --class_weights \
    --evaluation_strategy "epoch" \
    --learning_rate 0.0001 \
    --lr_schedule "cosine" \
    --disable_tqdm False \
    --run_name "ner-pubmedbert-labelsmoothing" \
    --do_train \
    --do_eval \
    --do_predict 

# Usig minmax_scale between 0.1 and 0.9 and reduction mean in the CrossEntropyLoss function we obtained

                precision    recall  f1-score   support

          CELL       0.69      0.84      0.76      4948
     EXP_ASSAY       0.52      0.72      0.60      9885
      GENEPROD       0.78      0.91      0.84     21869
      ORGANISM       0.66      0.92      0.77      3464
SMALL_MOLECULE       0.67      0.88      0.76      6431
   SUBCELLULAR       0.64      0.84      0.73      3850
        TISSUE       0.59      0.81      0.69      2975

     micro avg       0.68      0.86      0.76     53422
     macro avg       0.65      0.85      0.74     53422
  weighted avg       0.68      0.86      0.76     53422

{'test_loss': 0.2393004447221756, 'test_accuracy_score': 0.9390916097405456, 'test_precision': 0.677549387198601, 'test_recall': 0.8558084684212497, 'test_f1': 0.7563172565529906, 'test_runtime': 52.6869, 'test_s
amples_per_second': 156.111, 'test_steps_per_second': 0.626}

```

Adding class weights and label smoothing factor together

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --label_smoothing_factor 0.5 \
    --class_weights \
    --per_device_train_batch_size 16 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --learning_rate 0.0001 \
    --lr_schedule "cosine" \
    --disable_tqdm False \
    --run_name "ner-pubmedbert-labelsmoothing" \
    --do_train \
    --do_eval \
    --do_predict 

                precision    recall  f1-score   support

          CELL       0.68      0.82      0.74      4948
     EXP_ASSAY       0.55      0.68      0.61      9885
      GENEPROD       0.78      0.91      0.84     21869
      ORGANISM       0.70      0.91      0.79      3464
SMALL_MOLECULE       0.68      0.86      0.76      6431
   SUBCELLULAR       0.67      0.82      0.74      3850
        TISSUE       0.62      0.80      0.70      2975

     micro avg       0.69      0.84      0.76     53422
     macro avg       0.67      0.83      0.74     53422
  weighted avg       0.69      0.84      0.76     53422

{'test_loss': 0.20707711577415466, 'test_accuracy_score': 0.940903593401981, 'test_precision': 0.6915936764004126, 'test_recall': 0.8410018344502265, 'test_f1': 0.7590150779237235, 'test_runtime': 53.2697, 'test_
samples_per_second': 154.403, 'test_steps_per_second': 0.619}

```

The winnnigs are not so big. Let us try with the biolink large model to see if we actually win something compred to the 
published model.

```bsh
python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --disable_tqdm False \
    --label_smoothing_factor 0.5 \
    --class_weights \
    --per_device_train_batch_size 8 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --learning_rate 0.00005 \
    --lr_schedule "cosine" \
    --disable_tqdm False \
    --run_name "ner-biolinkbert-labelsmoothing" \
    --do_train \
    --do_eval \
    --do_predict 

               precision    recall  f1-score   support

          CELL       0.68      0.82      0.75      4948
     EXP_ASSAY       0.56      0.68      0.61      9885
      GENEPROD       0.79      0.91      0.84     21865
      ORGANISM       0.70      0.91      0.79      3464
SMALL_MOLECULE       0.71      0.86      0.77      6431
   SUBCELLULAR       0.67      0.82      0.74      3850
        TISSUE       0.63      0.82      0.71      2975

     micro avg       0.70      0.84      0.76     53418
     macro avg       0.68      0.83      0.75     53418
  weighted avg       0.70      0.84      0.76     53418

{'test_loss': 0.20438554883003235, 'test_accuracy_score': 0.9416225972282724, 'test_precision': 0.7003247665740249, 'test_recall': 0.839660788498259, 'test_f1': 0.7636893005516583, 'test_runtime': 87.9633, 'test_
samples_per_second': 93.505, 'test_steps_per_second': 0.375}

```