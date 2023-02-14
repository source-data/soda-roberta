#!/bin/bash
mkdir data/results/panelization

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
            "biobert-base-cased" )

large_model_list=(
            "michiyasunaga/BioLinkBERT-large" \
            "dmis-lab/biobert-large-cased-v1.1" \
            "EMBO/BioMegatron345mUncased" )
large_model_names=(
            "BioLinkBERT-large" \
            "biobert-large-cased" \
            "BioMegatron345mUncased" )

for i in ${!small_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task PANELIZATION \
        --from_pretrained ${small_model_list[$i]} \
        --per_device_train_batch_size 32 \
        --add_prefix_space \
        --learning_rate 0.0001 \
        --num_train_epochs 2.0 \
        --disable_tqdm False \
        --report_to none \
        --run_name "${small_model_names[$i]}_PANELIZATION" > "data/results/panelization/${small_model_names[$i]}.txt"
done

for i in ${!large_model_list[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task PANELIZATION \
        --from_pretrained ${large_model_list[$i]} \
        --per_device_train_batch_size 8 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --add_prefix_space \
        --learning_rate 0.00005 \
        --num_train_epochs 2.0 \
        --disable_tqdm False \
        --report_to none \
        --run_name "${large_model_names[$i]}_PANELIZATION" > "data/results/panelization/${large_model_names[$i]}.txt"
done
