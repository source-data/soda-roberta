# Presenting the SoDa dataset: Experiments in the paper

This document contains the commands or scripts run to generate the experiments
shown and mentioned in Abreu-Vicente & Lemberger 2022 (in prep). We take this approach
based on a strong believe on open science being the key to scientific progress.

**NOTE** The version of all the scripts is set to `--report_to none`. If you wish to 
see the metrics of the training you can change that line in any of the bash scripts.
You can visit the [🤗 library to check which reporting tools are allowed.](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to)

The models benchmarked in the paper are listed below:

* [`EMBO/bio-lm`](https://huggingface.co/EMBO/bio-lm)
* [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
* [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
* [`michiyasunaga/BioLinkBERT-base`](https://huggingface.co/michiyasunaga/BioLinkBERT-base)
* [`roberta-base`](https://huggingface.co/roberta-base)
* [`bert-base-cased`](https://huggingface.co/bert-base-cased)
* [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
* [`dmis-lab/biobert-base-cased-v1.2`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
* [`allenai/biomed_roberta_base`](https://huggingface.co/allenai/biomed_roberta_base)
* [`michiyasunaga/BioLinkBERT-large`](https://huggingface.co/michiyasunaga/BioLinkBERT-large)
* [`dmis-lab/biobert-large-cased-v1.1`](https://huggingface.codmis-lab/biobert-large-cased-v1.1)
* [`EMBO/BioMegatron345mUncased`](https://huggingface.co/EMBO/BioMegatron345mUncased)
* [`EMBO/BioMegatron345mCased`](https://huggingface.co/EMBO/BioMegatron345mCased)
* [`google/canine-s`](https://huggingface.co/google/canine-s)
* [`google/canine-c`](https://huggingface.co/google/canine-c)

The dataset used for this experiment is the [`EMBO/sd-nlp-non-tokenized`](https://huggingface.co/datasets/EMBO/sd-nlp-non-tokenized).

## 1. Panelization

From the base repository folder just call:

```bash
    sh bash_scripts/run_panelization.sh
```

The script will run through the 15 models of the paper, all except the `EMBO/bio-lm` model trained
on SMALL parts of text. The output of the process will be stored in text files in the folder
`$REPO_BASE/data/results/panelization/`. Note that if this folder does not exist, the 
script will complain about it. Just make sure you create the folder with:

```bash
    mkdir $REPO_HOME/data/results/panelization/
```

The panelization task can be visualized with the notebook `$REPO_HOME/notebooks/visualizing_panelization.ipynb`.

## 2. NER

### 2.1 Benchmarking multi-tasking NER

The script will run through the 15 models of the paper, all except the `EMBO/bio-lm` model trained
on SMALL parts of text. The output of the process will be stored in text files in the folder
`$REPO_HOME/data/results/ner/`. Note that if this folder does not exist, the 
script will complain about it. Just make sure you create the folder with:

```bash
    mkdir $REPO_HOME/data/results/ner/
```

```bash
    sh bash_scripts/run_ner_all_labels_benchmark.sh
```

### 2.2 Benchmarking multi-task vs. single-task NER

This experiment can be run using the bash command below.

```bash
    sh bash_scripts/run_ner_select_labels_benchmark.sh
```

In this case, we have decided to use the best model of our benchmark. Note that the best results are
obtained with `BioLink-large`. However, since `pubmed-bert` is very close in the results while having
roughly half of biolink large parameters, we use `pubmed-bert` in this experiment. It saves 
time and energy. 

We expect that the results obtained with `pubmed-bert` can be extrapolated to `BioLink-large` or any 
other model. If you wish to check this assumption, you can change 
`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` for `michiyasunaga/BioLinkBERT-large`
or any other model name in the bash script.

The results of this experiment will be stored in `$REPO_HOME/data/results/ner_single_task/`.

Note that there are 254 possible label permutations without repetition 

![equation](https://latex.codecogs.com/svg.image?P(n,r)&space;=&space;\sum_{r=1,N}&space;\frac{n!}{(n-r)!})

(N is the total number of labels in our dataset, 8, and r the number of labels used in a permutation)
on which we can train the model. Doing so in all different combinations is a highly resource consuming
work. 

Here we show the combinations used in the paper and the single task cases. It is up to the user to 
explore any other combination of labels.

### 2.3 Benchmarking models trained and fine-tuned in our dataset in other benchmarks

**To be implemented**

We will fine-tune our models in our dataset and then we will run the model doing
inference in the BLURB datasets. The idea here is to know how generalizable models trained in
our dataset would be.

```bash
    sh bash_scripts/run_soda_model_in_other_benchmarks.sh
```

## 3. Entity roles in experiments

We benchmark the performance of 13 different models into identifying 
entities such as `measured` or `assayed` components. This task is limited to the
`GENEPROD` and `SMALL_MOLECULE` entities. We build separate models for each.

Note that the process of the training masks the entities in the text before
proceeding to classify them as `measured` or `assayed`.

The experiment can be run using:

```bash
    sh bash_scripts/run_smallmol_roles.sh
    sh bash_scripts/run_geneprod_roles.sh
```

The results will be stored in:

```bash
    mkdir $REPO_HOME/data/results/smallmol_roles
    mkdir $REPO_HOME/data/results/geneprod_roles
```
