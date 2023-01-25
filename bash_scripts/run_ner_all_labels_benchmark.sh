#!/bin/bash

mkdir data/results/ner

small_model_list=("EMBO/bio-lm" \
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" \
            "michiyasunaga/BioLinkBERT-base" \
            "roberta-base" \
            "bert-base-cased" \
            "bert-base-uncased" \
            "dmis-lab/biobert-base-cased-v1.2" \
            "allenai/biomed_roberta_base")

small_model_names=("EMBO-bio-lm" \
            "PubMedBERT-base-uncased-abstract-fulltext" \
            "PubMedBERT-base-uncased-abstract" \
            "BioLinkBERT-base" \
            "roberta-base" \
            "bert-base-cased" \
            "bert-base-uncased" \
            "biobert-base-cased" \
            "biomed_roberta_base")

large_model_list=(
            "michiyasunaga/BioLinkBERT-large" \
            "dmis-lab/biobert-large-cased-v1.1" \
            "EMBO/BioMegatron345mUncased" \
            "EMBO/BioMegatron345mCased")
large_model_names=(
            "BioLinkBERT-large" \
            "biobert-large-cased" \
            "BioMegatron345mUncased" \
            "BioMegatron345mCased")

for i in ${!small_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained ${small_model_list[$i]} \
        --per_device_train_batch_size 16 \
        --add_prefix_space \
        --num_train_epochs 2.0 \
        --learning_rate 0.0001 \
        --disable_tqdm False \
        --report_to none \
        --ner_labels all \
        --do_train \
        --do_eval \
        --do_predict \
        --run_name "${small_model_names[$i]}_NER_all_labels" > "data/results/ner/${small_model_names[$i]}_ner_all_labels.txt"
done

for i in ${!large_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained ${large_model_list[$i]} \
        --per_device_train_batch_size 8 \
        --add_prefix_space \
        --num_train_epochs 2.0 \
        --learning_rate 0.00005 \
        --disable_tqdm False \
        --report_to none \
        --ner_labels all \
        --do_train \
        --do_eval \
        --do_predict \
        --run_name "${large_model_names[$i]}_NER_all_labels" > "data/results/ner/${large_model_names[$i]}_ner_all_labels.txt"
done

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-character-level-ner" \
    --task NER \
    --from_pretrained "google/canine-s" \
    --per_device_train_batch_size 4 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --report_to none \
    --gradient_accumulation_steps 2 \
    --ner_labels all \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "canine-s_NER_all_labels" > "data/results/ner/canine-s_ner_all_labels.txt"

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-character-level-ner" \
    --task NER \
    --from_pretrained "google/canine-c" \
    --per_device_train_batch_size 4 \
    --add_prefix_space \
    --num_train_epochs 2.0 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --report_to none \
    --gradient_accumulation_steps 2 \
    --ner_labels all \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name "canine-c_NER_all_labels" > "data/results/ner/canine-c_ner_all_labels.txt"

