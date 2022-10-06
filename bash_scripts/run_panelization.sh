#!/bin/bash

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
        --task PANELIZATION \
        --from_pretrained ${small_model_list[$i]} \
        --per_device_train_batch_size 16 \
        --add_prefix_space \
        --learning_rate 0.0001 \
        --num_train_epochs 2.0 \
        --disable_tqdm False \
        --report_to none \
        --do_train \
        --do_eval \
        --do_predict \
        --run_name "${small_model_names[$i]}_PANELIZATION" > "data/results/panelization/${small_model_names[$i]}.txt"
done

for i in ${!large_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task PANELIZATION \
        --from_pretrained ${large_model_list[$i]} \
        --per_device_train_batch_size 4 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --add_prefix_space \
        --learning_rate 0.00005 \
        --num_train_epochs 2.0 \
        --disable_tqdm False \
        --report_to none \
        --do_train \
        --do_eval \
        --do_predict \
        --run_name "${large_model_names[$i]}_PANELIZATION" > "data/results/panelization/${large_model_names[$i]}.txt"
done
