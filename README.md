
SoDa-RoBERTa
============

SODA-RoBERTa is a **So**urce **Da**ta resource for training __RoBERTa__ transformers for natural language processing tasks in cell and molecular biology.

SourceData database: https://sourcedata.io, "SourceData: a semantic platform for curating and searching figures"
Liechti R, George N, Götz L, El-Gebali S, Chasapi A, Crespo I, Xenarios I, Lemberger T, Nature Methods, https://doi.org/10.1038/nmeth.4471

RoBERTa transformer is a BERT derivative: https://huggingface.co/transformers/model_doc/roberta.html, "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

SODA-RoBERTa uses the huggingface (https://huggingface.co) and PyTorch (https://pytorch.org/) frameworks.

The models trained below are used in the SmartTag engine that tags biological entities and their experimental roles in figure legends. 

Tagging biological entities and classifying their role as measured vs controlled variables (i.e. target of controoled experimental interventions) allows to  derive a knowledge graph representing causal scientific hypotheses that are tested in specific experiments.

SmartTag uses a 3-step pipeline: 

1. Segmentation of the text of figure legends into sub-panel legends.
2. Named Entity Recognition of bioentities and experimental methods.
3. Semantic tagging of the experimental role of gene products and small molecules as measured variable or controlled variable.

Accordingly, a specialized language model for scientific biological language is fine tuned into 4 models for the respective tasks: PANELIZATION, NER, GENEPROD_ROLES and SMALL_MOL_ROLES. These models are based on the find-tuning of a language model trained on abstracts and figure legends of scientific articles available in PubMedCentral (http://europepmc.org/).

The datasetse and trained models are available at https://huggingface.co/EMBO.

We provide in `docs/` instructions to train the language model by fine tuning a pretrained Roberta transformer on text from PubMedCentral and by training the 4 specific token classification models using the SourceData datset. Training can be done useing the command line using the modules in `smtag.cli` or in jupyter notebooks (see [`training_protocol_LM.ipynb`](./training_protocol_LM.ipynb) and [`training_protocol_TOKCL.ipynb`](./training_protocol_TOKCL.ipynb) notebook).

The training raw data is in the form of XML files. SODA-ROBERTA provides tools to convert XML into tagged datasets that can be used for training transormers models. At inference stage, the tagged text is serialized back into json.


# Quick access to the pretrained SmartTag pipeline

Setup a Python virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

Install `docker` (https://docs.docker.com/engine/install/). 
Install`docker-compose` (https://docs.docker.com/compose/install/).

    pip install docker-compose==1.28.5

SmartTag can used with this command:

    docker-compose -f smtag.yml run --rm smtag "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."

This will pull automatically the docker image `tlemberger/smarttag:latest` from [dockerhub](https://hub.docker.com/).

The first time the image is run, the models and tokenizers will be downloaded automatically from https://huggingface.co/EMBO and cached in the Docker-managed volume `/cache`.

SmarTag can be included in separate projects via its Docker image:

    FROM tlemberger/smarttag:latest
    # rest of the project's Dockerfile

The tagger can be importer in python:

    from smtag.pipeline import SmartTagger
    smtg = SmartTagger()
    text = "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."
    tagged = smtg.tag(text)  # json output

Or via the command line:

    python -m smtag.cli.inference.tag "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."

To build a new image for smarttag for dockerhub user `anotheruser`:

    docker build -t anotheruser/smarttag:tagname -f DockerfileSmartTag  smtag

Push to dockerhub:

    docker login --username=anotheruser
    docker push anotheruser/smarttag:tagname 

# Quick overview

In a nutshell the following modules are involved in training and inference:

- `config` specifies application-wide preferences such as the type of model and tokenizer, exmaple lengths, etc...
- the sourcedata datasets is downloaded with `smartnode`
- examples are parsed from the xml with `extract`
- `dataprep` tokenizes examples and encodes (`encoder`) xml elements as labels with `xml2labels` maps
- `train` uses `loader` to load the dataset in the form expected by transformers and uses `datacollator` to generate batches and masks according to the task selected for training the model
- `tb_callback` customizes display of training and validation losses during traiing and `metrics` is run on the test set at the end of the training 
- `pipeline` integrates all the models in a single inference pipeline

Language modeling and token classification have their speclized training (`train_lm` vs `train_tokcl`) and loading (`loader_lm` vs `loader_tokcl`) modules.

Language modeling uses a task we call 'targeted masked language modeling', whereby specific part-of-speech tokens are masked probabilitically. The current configurations allow the following masking:
    - DET: determinant
    - VERBS: verbs
    - SMALL: any determinants, conjunctions, prepositions or pronouns


# Training

See the [`./training_protocol_TOKCL.ipynb`](./training_protocol_TOKCL.ipynb) Jupyter notebook or [`./docs/training.md`](/docs/training.md) on training the models.


To start the notebook

    tmux  # optional but practical
    docker-compose up -d
    docker-compose exec nlp bash
    jupyter notebook list  # -> Currently running servers: http://0.0.0.0:8888/?token=<tokenid>

# Posting dataset and models

See [`./docs/dataset_sharing.md`](./docs/dataset_sharing.md) on posting dataset and models.
