
SoDa-RoBERTa
============

SODA-RoBERTa is a **So**urce **Da**ta resource for training __RoBERTa__ transformers for natural language processing tasks in cell and molecular biology.

SourceData database: https://sourcedata.io, "SourceData: a semantic platform for curating and searching figures"
Liechti R, George N, Götz L, El-Gebali S, Chasapi A, Crespo I, Xenarios I, Lemberger T, Nature Methods, https://doi.org/10.1038/nmeth.4471

RoBERTa transformer is a BERT derivative: https://huggingface.co/transformers/model_doc/roberta.html, "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

SODA-RoBERTa uses the huggingface (https://huggingface.co) and PyTorch (https://pytorch.org/) frameworks.

The models trained below are used in the SmartTag engine that tags biological entities and their experimental roles in figure legends. Tagging biological entites that are the object of investigation in specific experiments reported in scientific figure and classifying their role as measured vs controlled variables allows to easily derive a knowledge graph representing scientific hypotheses reported in the literature.

SmartTag uses a 3-step pipeline: 

1. Segmentation of the text of figure legends into sub-panel legends.
2. Named Entity Recognition of bioentities and experimental methods.
3. Semantic tagging of the experimental role of generoducts as measured variable or controlled variable.

Accordingly, 3 models are trained with the respective tasks: PANELIZATION, NER, ROLES. We provide below the instructions on how to train these 3 models either using a specialized language model trained on biological text from PubMedCentral or using pre-trained Roberta transformers.

The training data is in the form of XML data. SODA-ROBERTA provides therefore tools to convert XML into tagged datasets that can be used for token classification tasks. At inference stage, the tagged text is serialized back into json or xml.

# Quick access to the pretrained SoDa-RoBERTa models and SmartTag pipeline.

Under construction.

# General Setup

Create the appropriate layout

```
mkdir lm_dataset #LM_DATASET
mkdir lm_model # LM_MODEL and TOKENIZER
mkdir ner_dataset  # NER_DATASET
mkdir tokcl_models  # NER_MODEL_PATH
mkdir cache  # HUGGINGFACE_CACHE
```

Edit .env.example accordingly and save as .env

Build and start the Docker container:

```
docker-compose build
docker-compose up -d
```

This will start both the lm service and a tensorboard service on port 6007, which allows to visualize training on https://localhost:6007

Start a tmux session (more convenient when training lasts a while) and run bash from the lm service:

```
tmux
docker-compose run --rm lm 
```

# Train specialized language model based on PubMed Central

## Setup

Download Open Access content from PubMed Central (takes a while!):

```
mkdir oapmc
cd oapmc
```

Connect to the EMBL-EBI ftp server:

```
ftp --no-prompt ftp.ebi.ac.uk 
# Name: anonymous
```

Navigate to to the PubMed Central Open Access subset directory and initiate the download:

```
cd /pub/databases/pmc/oa
mget *.xml.gz
exit
```

Expand the files:

```
gunzip *.gz
cd ..
```

The next steps need to be run form within the docker container:

```
tmux a # or tmux if ession does not exist
docker-compose run --rm lm bash
```

Extract articles from the JATS XML files but keep the XML so that sub-section of articles can be extracted later.

```
python -m common.extract oapmc oapmc_articles -P .//article --keep-xml
```

Randomly split aticles into train, eval, test sets. It is better to do this now, so that these subsets remain as independent as possible. It is important to do so when sevral examples (i.e. figure legends) are extracted per article otherwise accuracy metrics may be over optimistic (i.e. examples extracted from the same article should NOT be distributed across train, eval and test).

```
python -m common.split oapmc_articles
```

Extract text from the abstracts

```
python -m common.extracts oapmc_articles/train oapmc_abstracts/train -P .//abstract
python -m common.extracts oapmc_articles/eval oapmc_abstracts/eval -P .//abstract
python -m common.extracts oapmc_articles/test oapmc_abstracts/test -P .//abstract
```

## Train tokenizer

```
python -m lm.tokentrain oapmc_abstracts
```

## Tokenize and prepare dataset

python -m lm.dataprep oapmc_abstracts


## Train language model

```
python -m lm.train
```

# Fine tuning of pre-trained language model

## Setup

Download the SourceData raw dataset (xml files):

```
wget <url>
```

Split the original documents into train, eval and test sets. This is done at the document level since each document may contain several examples. Doing the split already now ensures more independent eval and test sets.

```
python -m common.split 191012/ -X xml
```

Extract the examples for NER using an XPAth that identifies individual panel legends within figure legends:

```
mkdir sourcedata
python -m common.extract 191012/train sourcedata/train -P .//sd-panel --keep-xml
python -m common.extract 191012/eval sourcedata/eval -P .//sd-panel --keep-xml
python -m common.extract 191012/test sourcedata/test -P .//sd-panel --keep-xml
```

Same thing but using a XPath for entire figure legends encompassing several panel legends. This will be used to learn segmentation of figure legends into panel legends:

```
mkdir panelization
python -m common.extract 191012/train panelization/train -P .//fig --keep-xml
python -m common.extract 191012/eval panelization/eval -P .//fig --keep-xml
python -m common.extract 191012/test panelization/test -P .//fig --keep-xml
```

Prepare the dataset for NER and ROLE labels:

```
python -m tokcl.dataprep sourcedata
```

## Train the models

Train the NER task to learn entity types:

```
python -m tokcl.train NER \
--output_dir=ner_model/NER \
--overwrite_output_dir \
--learning_rate=1e-5 \
--num_train_epochs=10 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--evaluation_strategy='steps' \
--save_total_limit=3 \
--logging_steps=20 \
--eval_steps=20 \
--save_steps=100
```


Train the ROLES task to learn entity roles:

```
python -m tokcl.train ROLES \
--output_dir=ner_model/ROLES \
--overwrite_output_dir \
--learning_rate=5e-5 \
--num_train_epochs=20 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--evaluation_strategy='steps' \
--save_total_limit=3 \
--logging_steps=20 \
--eval_steps=20 \
--save_steps=100
```

Prepare the dataset for the PANELIZATION task:

```
rm -fr ner_dataset  # dataprep does not overwrite to avoid disasters
python -m tokcl.dataprep panelization
```

Train the PANELIZATION task to learn panel segmentation:

```
python -m tokcl.train PANELIZATION \
--output_dir=ner_model/PANELIZATION \
--overwrite_output_dir \
--learning_rate=1e-5 \
--num_train_epochs=100 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--evaluation_strategy='steps' \
--save_total_limit=3 \
--logging_steps=50 \
--eval_steps=50 \
--save_steps=100
```

## Use the pipeline

Try smtag tagging:

```
python -m infer.smtag "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."
```



