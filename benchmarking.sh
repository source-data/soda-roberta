# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "EMBO/bio-lm" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "EMBO_bio-lm_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "roberta-base" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "roberta-base_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "bert-base-cased" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "bert-base-cased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "bert-base-uncased" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "bert-base-uncased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "dmis-lab/biobert-base-cased-v1.2" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "biobert-base-cased-v1_2_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10 --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "PubMedBERT-base-uncased-abstract_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#      --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --disable_tqdm True --hyperparameter_search \
#      --hp_experiment_name "PubMedBERT-base-uncased-abstract-fulltext_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "dmis-lab/biobert-large-cased-v1.1" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "biobert-large-cased-v1_1_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8
    
# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "roberta-large" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "roberta-large_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "bert-large-cased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "bert-large-cased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "bert-large-uncased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "bert-large-uncased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "michiyasunaga/BioLinkBERT-base" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "michiyasunaga___BioLinkBERT-base_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "michiyasunaga/BioLinkBERT-large" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "michiyasunaga___BioLinkBERT-large_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "EMBO/BioMegatron345mCased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "BioMegatron345mCased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10 --hp_tune_samples 6

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task SMALL_MOL_ROLES \
#     --from_pretrained "EMBO/BioMegatron345mUncased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "BioMegatron345mUncased_SMALLMOL" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 6

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task PANELIZATION \
    --from_pretrained "lm_models/lm_models_small/checkpoint-1125000/" \
    --disable_tqdm False \
    --masked_data_collator \
    --do_train \
    --do_eval \
    --do_predict 

python -m smtag.cli.tokcl.train \
    --loader_path "EMBO/sd-nlp" \
    --task PANELIZATION \
    --from_pretrained "EMBO/bio-lm" \
    --disable_tqdm True \
    --hyperparameter_search \
    --hp_experiment_name "EMBO_bio-lm_NER" \
    --hp_gpus_per_trial 4 \
    --hp_tune_samples 16 
