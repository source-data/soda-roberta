#!/bin/bash

mkdir data/results/ner

small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" \
            "michiyasunaga/BioLinkBERT-base" \
            "roberta-base" \
            "bert-base-cased" \
            "bert-base-uncased" \
            "dmis-lab/biobert-base-cased-v1.2" )

small_model_names=("PubMedBERT-base-uncased-abstract" \
            "BioLinkBERT-base" \
            "roberta-base" \
            "bert-base-cased" \
            "bert-base-uncased" \
            "biobert-base-cased")

large_model_list=(
            "michiyasunaga/BioLinkBERT-large" \
            "dmis-lab/biobert-large-cased-v1.1" \
            "EMBO/BioMegatron345mUncased")
large_model_names=(
            "BioLinkBERT-large" \
            "biobert-large-cased" \
            "BioMegatron345mUncased")

# for i in ${!small_model_list[@]}; do
#     python -m smtag.cli.tokcl.train \
#         --loader_path "EMBO/sd-nlp-non-tokenized" \
#         --task NER \
#         --from_pretrained ${small_model_list[$i]} \
#         --per_device_train_batch_size 16 \
#         --add_prefix_space \
#         --num_train_epochs 2.0 \
#         --learning_rate 0.0001 \
#         --disable_tqdm False \
#         --report_to none \
#         --ner_labels all \
#         --run_name "${small_model_names[$i]}_NER_all_labels" > "data/results/ner/${small_model_names[$i]}_ner_all_labels.txt"
# done

# for i in ${!large_model_list[@]}; do
#     python -m smtag.cli.tokcl.train \
#         --loader_path "EMBO/sd-nlp-non-tokenized" \
#         --task NER \
#         --from_pretrained ${large_model_list[$i]} \
#         --per_device_train_batch_size 8 \
#         --add_prefix_space \
#         --num_train_epochs 2.0 \
#         --learning_rate 0.00005 \
#         --disable_tqdm False \
#         --report_to none \
#         --ner_labels all \
#         --run_name "${large_model_names[$i]}_NER_all_labels" > "data/results/ner/${large_model_names[$i]}_ner_all_labels.txt"
# done

# for i in ${!small_model_list[@]}; do
#     python -m smtag.cli.tokcl.train \
#         --loader_path "EMBO/sd-nlp-non-tokenized" \
#         --task NER \
#         --from_pretrained ${small_model_list[$i]} \
#         --use_crf \
#         --per_device_train_batch_size 16 \
#         --add_prefix_space \
#         --num_train_epochs 2.0 \
#         --learning_rate 0.0001 \
#         --disable_tqdm False \
#         --report_to none \
#         --ner_labels all \
#         --run_name "${small_model_names[$i]}_NER_all_labels_CRF" > "data/results/ner/${small_model_names[$i]}_ner_all_labels_CRF.txt"
# done

for i in ${!large_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained ${large_model_list[$i]} \
        --use_crf \
        --per_device_train_batch_size 8 \
        --add_prefix_space \
        --num_train_epochs 2.0 \
        --learning_rate 0.00005 \
        --disable_tqdm False \
        --report_to none \
        --ner_labels all \
        --run_name "${large_model_names[$i]}_NER_all_labels_CRF" > "data/results/ner/${large_model_names[$i]}_ner_all_labels_CRF.txt"
done



python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" \
    --use_crf \
    --per_device_train_batch_size 8 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.00005 \
    --disable_tqdm False \
    --report_to none \
    --ner_labels all \
    --max_steps 50