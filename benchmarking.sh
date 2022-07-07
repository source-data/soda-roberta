# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "EMBO/bio-lm" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "EMBO_bio-lm_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "roberta-base" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "roberta-base_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "bert-base-cased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "bert-base-cased_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "bert-base-uncased" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "bert-base-uncased_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "dmis-lab/biobert-base-cased-v1.2" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "biobert-base-cased-v1_2_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10 --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "PubMedBERT-base-uncased-abstract_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

# python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
#     --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --disable_tqdm True --hyperparameter_search \
#     --hp_experiment_name "PubMedBERT-base-uncased-abstract-fulltext_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "dmis-lab/biobert-large-cased-v1.1" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "biobert-large-cased-v1_1_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8
    
python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "roberta-large" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "roberta-large_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "bert-large-cased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-large-cased_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "bert-large-uncased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-large-uncased_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-base" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "michiyasunaga___BioLinkBERT-base_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 12

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "michiyasunaga/BioLinkBERT-large" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "michiyasunaga___BioLinkBERT-large_GENEPROD_ROLES" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 8

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "EMBO/BioMegatron345mCased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "BioMegatron345mCased_NER" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10 --hp_tune_samples 6

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task GENEPROD_ROLES \
    --from_pretrained "EMBO/BioMegatron345mUncased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "BioMegatron345mUncased_NER" --hp_gpus_per_trial 1 --hp_cpus_per_trial 10  --hp_tune_samples 6
