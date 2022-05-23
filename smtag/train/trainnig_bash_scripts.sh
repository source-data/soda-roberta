python -m smtag.cli.tokcl.train EMBO/sd-nlp-non-tokenized NER \
                                --from_pretrained bert-base-cased \
                                --model_type Autoencoder \
                                --masked_data_collator False \
                                --tokenizer bert-base-cased \
                                --do_predict True \
                                --do_test True \
                                --dropout 0.2 \
                                --hidden_size_multiple 64 \
                                --prediction_loss_only False \
                                --per_device_train_batch_size 32\
                                --per_device_eval_batch_size 32\
                                --evaluation_strategy epoch \
                                --eval_steps 1 \
                                --label_smoothing_factor 0.0 \
                                --learning_rate 0.005 \
                                --num_train_epochs 1 \
                                --lr_scheduler_type linear \
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




