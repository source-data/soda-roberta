
SoDa-RoBERTa
============

SODA-RoBERTa is a **So**urce **Da**ta resource for training __RoBERTa__ transformers for natural language processing tasks in cell and molecular biology.

SourceData database: https://sourcedata.io, "SourceData: a semantic platform for curating and searching figures"
Liechti R, George N, Götz L, El-Gebali S, Chasapi A, Crespo I, Xenarios I, Lemberger T, Nature Methods, https://doi.org/10.1038/nmeth.4471

RoBERTa transformer is a BERT derivative: https://huggingface.co/transformers/model_doc/roberta.html, "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

SODA-RoBERTa uses the huggingface (https://huggingface.co) and PyTorch (https://pytorch.org/) frameworks.

The models trained below are used in the SmartTag engine that tags biological entities and their experimental roles in figure legends. 

Tagging biological entities that are the object of investigation in specific experiments reported in scientific figure and classifying their role as measured vs controlled variables allows to easily derive a knowledge graph representing scientific hypotheses reported in the literature. 

SmartTag uses a 3-step pipeline: 

1. Segmentation of the text of figure legends into sub-panel legends.
2. Named Entity Recognition of bioentities and experimental methods.
3. Semantic tagging of the experimental role of generoducts as measured variable or controlled variable.

Accordingly, a specialized language model for scientific biological language is fine tuned into 3 models for the respective tasks: PANELIZATION, NER, ROLES. These models are based on the find-tuning of a language model trained on abstracts and figure legends of scientific articles available in PubMedCentral (http://europepmc.org/).

The datasetse and trained models are available as such at https://huggingface.co/EMBO.

We provide below the instructions to train the language model by fine tuning a pretrained Roberta transformer on text from PubMedCentral and by training the 3 specific models using the SourceData datset.

The training data is in the form of XML files. SODA-ROBERTA provides therefore tools to convert XML into tagged datasets that can be used for token classification tasks. At inference stage, the tagged text is serialized back into json or xml.

# Docker

Setup a Python virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

Install `docker` (https://docs.docker.com/engine/install/). 
Install`docker-compose` (https://docs.docker.com/compose/install/).

    pip install docker-compose==1.28.5

# Quick access to the pretrained SmartTag pipeline

SmartTag can used with this command:

    docker-compose -f smtag.yml run --rm smtag "This is an interesting brain."

This will pull automatically the docker image `tlemberger/smarttag:latest` from [dockerhub](https://hub.docker.com/).

The first time the image is run, the models and tokenizers will be downloaded automatically and cached in the Docker-managed volume `cache`.

To build a new image for smarttag for dockerhub user `anotheruser`:

    docker build -t anotheruser/smarttag:tagname -f src/predict/Dockerfile src

Push to dockerub:

    docker login --username=anotheruser
    docker push anotheruser/smarttag:tagname 

# General setup for training

Install `git-lfs` (https://git-lfs.github.com/).

Build and start the Docker container with docker-compose:

    docker-compose build
    docker-compose run --rm nlp

This will start the following services:

- `nlp` service to run the training commands (with bash as entrypoint)
- a `tensorboard` service on port 6007 (visita at http://localhost:6007) to visualize trainig
- `celery`/`rabbitmq`/`flower` to paralellize data preparation; progress can be followed at on https://localhost:5555 with flower running on port 5555.

The entrypoint in nlp is bash.

Within the container, the assumed default layout is as follows:


    /app
    /data
        ├── /xml         # XML files used as source of examples or for token classification
        │── /text        # text files used for languag modeling
        └── /json        # pre-tokenized and labeled datasets
    /lm_models           # language models
    /tockl_models        # token classification models
    /cache               # the cache used by the data loader


# Fine tuning a specialized language model based on PubMed Central

## Setup

Download Open Access content from PubMed Central (takes a while!):

    mkdir /data/xml/oapmc
    cd /data/xml/oapmc


Connect to the EMBL-EBI ftp server:

    ftp --no-prompt ftp.ebi.ac.uk 
    # Name: anonymous


Navigate to to the PubMed Central Open Access subset directory and initiate the download:

    cd /pub/databases/pmc/oa
    mget *.xml.gz
    exit


Expand the files:

    gunzip *.gz

Extract articles from the JATS XML files but keep the XML so that sub-sections (eg figures or abstracts) can be extracted later.

    python -m common.extract /data/xml/oapmc /data/xml/oapmc_articles -P .//article --keep-xml

Randomly split aticles into train, eval, test sets. 

Note: it is better to do this now rather than later, so that these subsets remain as independent as possible. It is important to do so when several examples (i.e. figure legends) are extracted per xml document, otherwise accuracy metrics may be over optimistic (i.e. examples extracted from the same article should NOT be distributed across train, eval and test).

    python -m common.split /data/xml/oapmc_articles

Extract text from the abstracts

    python -m common.extract /data/oapmc_articles  -P .//abstract
    # creates by default /data/text/oapmc_abstracts/ with train valid and test sub dir

Note: it is possibel to combined several XPath with the `|` operator to extract several kinds of elements. For example, extract both abstracts and figure legends (this would be very large):

    python -m common.extract /data/xml/oapmc_articles/ /data/text/oapmc_abstracts_figs/ --proba=0.5 --xpath ".//abstract | .//fig"

Note: the option --proba allows to determine the probability with wich each example is actually includedd; this is useful when the dataset is huge and only a random subset needs to be selected.

## Tokenize and prepare dataset

By default, the configuration file `common.config` specifies the pretrained 'roberta-base' model as statring point for fine tuning the language model. The appropriate tokenizer will also be used. To train a language model from scratch, set from_pretrained = '' in common.config and train the tokenizer:

    python -m lm.tokentrain /data/text/oapmc_abstracts  # ONLY WHEN TRAINING CUSTOMIZED MODEL!

Tokenized the data:

    python -m lm.dataprep /data/text/oapmc_abstracts /data/json/oapmc_abstracts

This can take a while for large datasets. To follow the progress, visit http://localhost:5555 (via the celery flower service)

## Train language model

    python -m lm.train /data/json/oapmc_abstracts

Note:   default, the model is saved in /lm_models so that it can be used for subsequent fine tuning for token classification

## Try it:

    python -m lm.try_it "A kinase phosphorylates its <mask> to activate it."

# Fine tuning of pre-trained language model

Now that we have a language model, we fine tune for token classification and text segmentation based on the SourceData dataset.

## Setup

Download the SourceData raw dataset (xml files):

    wget <url>
    unzip download.zip
    mv download /data/xml/sourcedata

    wget <url>
    unzip panelization_compendium.zip
    mv panelization_compendium /data/xml/panelization_compendium

Split the original documents into train, eval and test sets. This is done at the document level since each document may contain several examples. Doing the split already now ensures more independent eval and test sets.

    python -m common.split /data/xml/sourcedata/ -X xml
    python -m common.split /data/xml/panelization_compendium --extension xml

Extract the examples for NER and ROLES using an XPAth that identifies individual panel legends within figure legends:

    mkdir /data/xml/sd_panels
    python -m common.extract /data/xml/sourcedata /data/xml/sd_panels -P .//sd-panel --keep_xml

Using an XPath for entire figure legends encompassing several panel legends. This will be used to learn segmentation of figure legends into panel legends:

    mkdir /data/xml/sd_panelization
    python -m common.extract /data/xml/panelization_compendium /data/xml/sd_panelization --xpath .//fig --keep_xml

Prepare the datasets for NER, ROLES and PANELIZATION:

    mkdir /data/json/sd_panels
    python -m tokcl.dataprep /data/xml/sd_panels /data/json/sd_panels
    mkdir /data/json/sd_panelization
    python -m tokcl.dataprep /data/xml/sd_panelization /data/json/sd_panelization

## Post the datasets

Clone the EMBO/sd-nlp repo if not yet done:

    git clone https://huggingface.co/datasets/EMBO/sd-nlp
    cd sd-nlp

Install datasets and transformers

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install datasets
    pip install transformers

Copy the relevant dataset files

    cp -r ../soda-roberta/data/json/sd_panels.zip ./
    cp -r ../soda-roberta/data/json/sd_figs.zip ./

Update the dummy files.

Copy the loading script:

    cp ../soda-roberta/src/tokcl/loader.py sd-nlp.py

Make sure .zip files are tracked as lfs objects:

    git lfs install
    git lfs track .git
    git add .gitattributes
    commit -m "zip as lfs"

Add and commit files:

    git add sd_panels.zip
    git add sd_figs.zip
    git add dummy
    git commit -m "updating data files"
    git push



## Train the models

Train the NER task to learn entity types:

    python -m tokcl.train NER

Train the ROLES task to learn entity roles:

    python -m tokcl.train ROLES

Train the PANELIZATION task to learn panel segmentation:

    python -m tokcl.train PANELIZATION

## Post the models



## Use the pipeline

Tada! The trained models are now in /tokcl_models and you can try SmartTag-ging:

    python -m predict.smtag "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."

