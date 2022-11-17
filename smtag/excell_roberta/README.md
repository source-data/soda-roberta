# EMBO eXcellence in life sciences RoBERTa language model

This module contains the implementation of the 
EXcell-RoBERTa language model presented in Abreu-Vicente & Lemberger 2023b (in prep). 
The pre-training weights of this model will be available in HuggingFace, as well
as the datasets used.

In order to use this model, the user should first install the SODA-RoBERTa
package as explained in the main README file of this repository.

The model generation is done on a series of steps:

## Step 1: Tokenization

This is the first step to generate a language model from scratch. The tokenization
of EXcell-RoBERTa is a BPE tokenizer, similar to RoBERTa, but with the addition of
a NFKC normalizing step to normalize unicode characters. 

This is specially useful for greek letters. The tokenizer can be generated with the 
following command, given that a text file with the text corpus exists and is available.

```bash
    python -m smtag.excell_roberta.create_tokenizer \
        /app/data/text/oapmc_figs_full_text_pmc_for_tokenizer.txt \
        excell-roberta-tokenizer \
        --vocab_size 52000 \
        --min_freq 50
```

## Step 2: Definition of the model architecture

We provide in the code the parameters used and reported in 
Abreu-Vicente & Lemberger 2023b (in prep).

Any other combination of parameters will need to be modified by the
user.

## Step 3: Data preprocessing

This step is included on the script to generate the model. It accepts `jsonl` files with `['text']`
as field containing the strings for input.

## Step 4: Model training

The first training can be done using:

```shell
    python -m smtag.excell_roberta.model \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        --loglevel info \
        --output_dir "excell-roberta-lm"
```

Successive trainings can be done from given checkpoints of the model using:

```shell
    python -m smtag.excell_roberta.model \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        from_checkpoint 100000 \
        --loglevel info \
        --output_dir "excell-roberta-lm"
```

## Step 5: Model evaluation with NER in Source Data

TOnce the language model is trained, it will be automatically fine tune in the 
[EMBO/sd-nlp-non-tokenized](https://huggingface.co/datasets/EMBO/sd-nlp-non-tokenized)
dataset to see its performance on the NER task.

# Step 6: Model benchmarking



**NOTE** The version of all the scripts is set to `--report_to none`. If you wish to 
see the metrics of the training you can change that line in any of the bash scripts.
You can visit the [🤗 library to check which reporting tools are allowed.](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to)

The models benchmarked in the paper are listed below:

* [`EMBO/bio-lm`](https://huggingface.co/EMBO/bio-lm)
* [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
* [`dmis-lab/biobert-base-cased-v1.2`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
* [`allenai/biomed_roberta_base`](https://huggingface.co/allenai/biomed_roberta_base)

