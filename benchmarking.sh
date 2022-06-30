python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "EMBO/bio-lm" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "EMBO_bio-lm_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "roberta-base" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "roberta-base_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "bert-base-cased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-base-cased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "bert-base-uncased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-base-uncased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "dmis-lab/biobert-base-cased-v1.2" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "biobert-base-cased-v1_2_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "PubMedBERT-base-uncased-abstract_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "PubMedBERT-base-uncased-abstract-fulltext_NER" --hp_gpus_per_trial 1 --hp_tune_samples 24

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "dmis-lab/biobert-large-cased-v1.1" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "biobert-large-cased-v1_1_NER" --hp_gpus_per_trial 1 --hp_tune_samples 12
    
python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "roberta-large" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "roberta-large_NER" --hp_gpus_per_trial 1 --hp_tune_samples 12

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "bert-large-cased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-large-cased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 12

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "bert-large-uncased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "bert-large-uncased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 12

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "EMBO/BioMegatron345mCased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "BioMegatron345mCased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 8

python -m smtag.cli.tokcl.train --loader_path "EMBO/sd-nlp-non-tokenized" --task NER \
    --from_pretrained "EMBO/BioMegatron345mUncased" --disable_tqdm True --hyperparameter_search \
    --hp_experiment_name "BioMegatron345mUncased_NER" --hp_gpus_per_trial 1 --hp_tune_samples 8