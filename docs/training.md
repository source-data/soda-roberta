# General setup before training

Update the `.env.example` and save as `.env`. Recommended:

    LM_MODEL_PATH="/lm_models"
    TOKCL_MODEL_PATH="/tokcl_models"
    CACHE="/cache"
    RUNS_DIR="/runs"
    DUMMY_DIR="/dummy"

Install `git-lfs` (https://git-lfs.github.com/).

Build and start the Docker container with docker-compose:

    docker-compose build
    docker-compose up -d

This will start the following services:

- `nlp` to run the training commands (with bash as entrypoint); the default command opens a jupyter notebook on port 8888.
- `tensorboard` on port 6007 (visita at http://localhost:6007) to visualize trainig
- `celery`/`rabbitmq`/`flower` to paralellize data preparation; progress can be followed at on https://localhost:5555 with flower running on port 5555.

To use the command line within the container:

    docker-compose exec nlp bash


To obtain the token_id for the jupyter notebook:

    docker-compose exec nlp bash
    jupyter notebook list


Within the container, the assumed default layout is as follows:


    /app
    /data
        ├── /xml         # XML files used as source of examples
        │── /text        # text files with extracted examples (1 example per line)
        └── /json        # pre-tokenized and labeled datasets (1 example per line)
    /lm_models           # language models
    /tockl_models        # token classification models
    /cache               # the cache used by the data loader
    /dummy               # dummy dir when posting datasets

## config and `.env`

The location where models are save is determined in `.env`. Modify `.env.example` and save as `.env` before starting. A typical setup is:

    LM_MODEL_PATH="/lm_models"
    TOKCL_MODEL_PATH="/tokcl_models"
    CACHE="/cache"
    RUNS_DIR="/runs"
    DUMMY_DIR="/dummy"


Application-wide preferences are set in `config.py`.


# Fine tuning a specialized language model based on PubMed Central

## Downloading the Open Access corpus from EuropePMC

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

## Extraction of individual articles 

Extract articles from the JATS XML files but keep the XML so that sub-sections (eg figures or abstracts) can be extracted later.

    python -m smtag.cli.lm.extract /data/xml/oapmc /data/xml/oapmc_articles --xpath .//article --keep-xml

Randomly split aticles into train, eval, test sets. 

Note: it is better to do this now rather than later, so that these subsets remain as independent as possible. It is important to do so when several examples (i.e. figure legends) are extracted per xml document, otherwise accuracy metrics may be over optimistic (i.e. examples extracted from the same article should NOT be distributed across train, eval and test).

    python -m smtag.cli.prepro.split /data/xml/oapmc_articles

Extract text from the abstracts:

    python -m smtag.cli.lm.extract /data/oapmc_articles  --xpath .//abstract
    # creates by default /data/text/oapmc_abstracts/ with train valid and test sub dir

Note: it is possible to combine several XPath expressions with the `|` operator to extract several kinds of elements. For example, extract both abstracts and figure legends (this would be very large):

    python -m smtag.cli.lm.extract \
    /data/xml/oapmc_articles/ \
    /data/text/oapmc_abstracts_figs/ \
    --xpath ".//abstract | .//fig" \
    --inclusion_probability=0.5 

Note: the option --inclusion_probability allows to determine the probability with wich each example is actually included; this is useful when the dataset is huge and only a random subset is needed.


## Tokenize and prepare dataset

By default, the module `config.py` specifies the pretrained 'roberta-base' model as starting point for fine tuning the language model. The appropriate tokenizer will also be used.

Tokenize the data:

    python -m smtag.cli.lm.dataprep /data/text/oapmc_abstracts /data/json/oapmc_abstracts

This can take a while for large datasets. To follow the progress, visit http://localhost:5555 (via the celery flower service)

## Train language model

Four tasks (or data configurations) are supported:

- `MLM`: masked language modeling
- `DET`: part-of-speeach masking of determniants
- `VERB`: part-of-speeach masking of verbs
- `SMALL`: part-of-speeach masking of determinants, conjunctions, prepositions and pronouns

To train a conventional masked language model:

    python -m smtag.cli.lm.train \
    smtag.loader.loader_lm.py \
    MLM \
    --data_dir /data/json/oapmc_abstracts

Note:   default, the model is saved in /lm_models so that it can be used for subsequent fine tuning for token classification


# Fine tuning of pre-trained language model

Now that we have a language model, we fine tune for token classification and text segmentation based on the SourceData dataset.

## Setup

Download the SourceData raw dataset:

    wget <url>
    unzip download.zip
    mv download /data/xml/sourcedata

    wget <url>
    unzip panelization_compendium.zip
    mv panelization_compendium /data/xml/panelization_compendium

Note that the latest dataset can be prepared from the SourceData REST API using:

    python -m smtag.cli.prepro.get_sd  # takes a very long time!!

Split the original documents into train, eval and test sets. This is done at the document level since each document may contain several examples. Doing the split already now ensures more independent eval and test sets.

    python -m smtag.cli.prepro.split /data/xml/sourcedata/ --extension xml
    python -m smtag.cli.prepro.split /data/xml/panelization_compendium --extension xml

Extract the examples for NER and ROLES using an XPAth that identifies individual panel legends within figure legends:

    mkdir /data/xml/sd_panels
    python -m smtag.cli.prepro.extract /data/xml/sourcedata /data/xml/sd_panels -P .//sd-panel --keep_xml

Using an XPath for entire figure legends encompassing several panel legends. This will be used to learn segmentation of figure legends into panel legends:

    mkdir /data/xml/sd_panelization
    python -m smtag.cli.prepro.extract /data/xml/panelization_compendium /data/xml/sd_panelization --xpath .//fig --keep_xml

Prepare the datasets for NER, ROLES and PANELIZATION:

    mkdir /data/json/sd_panels
    python -m tokcl.dataprep /data/xml/sd_panels /data/json/sd_panels
    mkdir /data/json/sd_panelization
    python -m tokcl.dataprep /data/xml/sd_panelization /data/json/sd_panelization

The previous code will generate the data with the special `roberta-base` tokenization. 
That means that the data can be used **only** for models that have been pre-trained
using the `roberta-base` tokenizer. To generate a more general text that can be tokenized
with any tokenizer the following commands can be used.

    mkdir /data/json/sd_panels_general_tokenization
    python -m smtag.cli.tokcl.generatlTOKCL /data/xml/sd_panels /data/json/sd_panels_general_tokenization
    mkdir /data/json/sd_panelization_general_tokenization
    python -m smtag.cli.tokcl.generatlTOKCL /data/xml/sd_panelization /data/json/sd_panelization_general_tokenization

The resulting dataset will contain the text split into words, being each word correctly
labeled.

## Train the models

The `roberta-base` tokenize models can be trained using the following command, where
`TASK_NAME` must be replaced by one of the following 
`["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"]`

    python -m smtag.cli.tokcl.train TASK_NAME
