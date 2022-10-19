#!/bin/bash

model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

labels=( \
        "GENEPROD" \
        "TISSUE" \
        "ORGANISM" \
        "SMALL_MOLECULE" \
        "EXP_ASSAY" \
        "CELL" \
        "SUBCELLULAR" \
    )

mkdir data/results/ner_single_task

for i in ${!labels[@]}; do
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained $model \
        --per_device_train_batch_size 32 \
        --add_prefix_space \
        --num_train_epochs 2.0 \
        --learning_rate 0.0001 \
        --disable_tqdm False \
        --do_train \
        --do_eval \
        --do_predict \
        --report_to none \
        --ner_labels ${labels[$i]} \
        --run_name "${labels[$i]}_NER_single_task" > "data/results/ner_single_task/${labels[$i]}_ner.txt"
done
