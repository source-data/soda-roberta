#!/bin/bash

model="michiyasunaga/BioLinkBERT-large"

labels=( \
        "GENEPROD" \
        "TISSUE" \
        "ORGANISM" \
        "SMALL_MOLECULE" \
        "EXP_ASSAY" \
        "CELL" \
        "SUBCELLULAR" \
        "DISEASE" \
    )

small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
small_model_names=("PubMedBERT-base-uncased-abstract")
large_model_list=("michiyasunaga/BioLinkBERT-large")
large_model_names=("BioLinkBERT-large")

mkdir data/results/ner_single_task

for i in ${!labels[@]}; do
    for j in ${!small_model_list[@]}; do
        python -m smtag.cli.tokcl.train \
            --loader_path "EMBO/sd-nlp-non-tokenized" \
            --task NER \
            --from_pretrained ${small_model_list[$j]} \
            --per_device_train_batch_size 16 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.0001 \
            --disable_tqdm False \
            --report_to none \
            --ner_labels ${labels[$i]} \
            --run_name "${small_model_names[$j]}_${labels[$i]}_NER_single_task" > "data/results/ner_single_task/${small_model_names[$j]}_${labels[$i]}_ner.txt"
    done
done


for i in ${!labels[@]}; do
    for j in ${!large_model_list[@]}; do
        python -m smtag.cli.tokcl.train \
            --loader_path "EMBO/sd-nlp-non-tokenized" \
            --task NER \
            --from_pretrained ${small_model_list[$j]} \
            --per_device_train_batch_size 8 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.00005 \
            --disable_tqdm False \
            --report_to none \
            --ner_labels all \
            --report_to none \
            --ner_labels ${labels[$i]} \
            --run_name "${large_model_names[$j]}_${labels[$i]}_NER_single_task" > "data/results/ner_single_task/${large_model_names[$j]}_${labels[$i]}_ner.txt"
    done
done


for i in ${!labels[@]}; do
    for j in ${!small_model_list[@]}; do
        python -m smtag.cli.tokcl.train \
            --loader_path "EMBO/sd-nlp-non-tokenized" \
            --task NER \
            --from_pretrained ${small_model_list[$j]} \
            --use_crf \
            --per_device_train_batch_size 16 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.0001 \
            --disable_tqdm False \
            --report_to none \
            --ner_labels ${labels[$i]} \
            --run_name "${small_model_names[$j]}_${labels[$i]}_NER_single_task_CRF" > "data/results/ner_single_task/${small_model_names[$j]}_${labels[$i]}_ner_CRF.txt"
    done
done


for i in ${!labels[@]}; do
    for j in ${!large_model_list[@]}; do
        python -m smtag.cli.tokcl.train \
            --loader_path "EMBO/sd-nlp-non-tokenized" \
            --task NER \
            --from_pretrained ${small_model_list[$j]} \
            --use_crf \
            --per_device_train_batch_size 8 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.00005 \
            --disable_tqdm False \
            --report_to none \
            --ner_labels all \
            --report_to none \
            --ner_labels ${labels[$i]} \
            --run_name "${large_model_names[$j]}_${labels[$i]}_NER_single_task_CRF" > "data/results/ner_single_task/${large_model_names[$j]}_${labels[$i]}_ner_CRF.txt"
    done
done
