python -m smtag.cli.tokcl.train EMBO/sd-nlp-non-tokenized NER \
                                --from_pretrained bert-base-cased \
                                --model_type Autoencoder \
                                --masked_data_collator True \
                                --tokenizer bert-base-cased \
                                --do_predict True \
                                --prediction_loss_only False \
                                --per_device_train_batch_size 16\
                                --per_device_eval_batch_size 32\
                                --evaluation_strategy steps \
                                --eval_steps 1000 \
                                --label_smoothing_factor 0.0 \
                                --learning_rate 0.0005 \
                                --num_train_epochs 5 \
                                --lr_scheduler_type linear \
                                --save_strategy steps \
                                --save_steps 1000 \
                                --seed 42 \
                                --load_best_model_at_end True \
                                --metric_for_best_model f1 \
                                --greater_is_better True \
                                --report_to tensorboard \
                                --push_to_hub True \
                                --hub_strategy checkpoint \
                                --hub_model_id EMBO/bert-base-cased_NER-task \
                                --hub_token ${HUB_TOKEN} \
                                --adafactor False \
                                --adam_beta1 0.9 \
                                --adam_beta2 0.999 \
                                --adam_epsilon 1e-08 \
                                --weight_decay 0.0 \
                                --overwrite_output_dir \
                                --output_dir ./tokcl_models/EMBO_bert-base-cased_NER-task





