# General setup before training

Update the `.env.example` and save as `.env`. Recommended:

    LM_MODEL_PATH="/lm_models"
    TOKCL_MODEL_PATH="/tokcl_models"
    CACHE="/cache"
    RUNS_DIR="/runs"
    RUNTIME=""

Specify `RUNTIME="nvidia"` on a NVIDIA DGX GPU station.

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

Randomly split aticles into train, eval, test sets.

Note: it is better to do this now rather than later, so that these subsets remain as independent as possible. It is important to do so when several examples (i.e. figure legends) are extracted per xml document, otherwise accuracy metrics may be over optimistic (i.e. examples extracted from the same article should NOT be distributed across train, eval and test).

    python -m smtag.cli.prepro.split /data/xml/oapmc

Extract articles from the JATS XML files but keep the XML so that sub-sections (eg figures or abstracts) can be extracted later.

    python -m smtag.cli.prepro.extract /data/xml/oapmc /data/xml/oapmc_articles --xpath .//article --keep_xml

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
    loader/loader_lm.py \
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

    python -m smtag.cli.prepro.extract /data/xml/sourcedata /data/xml/sd_panels -P .//sd-panel --keep_xml

Using an XPath for entire figure legends encompassing several panel legends. This will be used to learn segmentation of figure legends into panel legends:

    python -m smtag.cli.prepro.extract /data/xml/sourcedata /data/xml/sd_panelization --xpath .//fig --keep_xml

Prepare the datasets for NER, ROLES and PANELIZATION. This step generates the data. This step will generate `JSONlines` files for the `train`, `test`, and `eval` splits. They will be word pre-tokenized and wth IOB labels for each word. This way of organizing the data is compatible with the best practices for token classification in the 🤗 framework. Note that it is important at this point to have the `config.py` file properly configured.

    mkdir /data/json/sd_panels
    python -m smtag.cli.tokcl.dataprep /data/xml/sd_panels /data/json/sd_panels
    mkdir /data/json/sd_panelization
    python -m tokcl.dataprep /data/xml/sd_panelization /data/json/sd_panelization

## Train the models

Single training for `token classification` tasks can be done using examples similar to the ones set below.
The default training will be generated by the following command. Changing between the different tasks can be donw by adding the argument to the attribute `--task`. One can be chosen between: 
`["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING", "PANELIZATION"]`. The first is an example of how to fine tune the `roberta-base` model. The second example shows how to fine-tune the source data language model.

```shell
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained roberta-base \
        --masked_data_collator \
        --disable_tqdm False \
        --do_train \
        --do_eval \
        --do_predict

    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained "EMBO/bio-lm" \
        --masked_data_collator \
        --disable_tqdm False \
        --do_train \
        --do_eval \
        --do_predict
```

There are two options two modify the arguments of the training. They can be specified in the batch itself. Alternatively, the default arguments can be edited in `data_classes.TrainingArgumentsTOKCL` and then run the scripts as they are shown above.

## Hyperparameter search

Hyperparameter search can also be generated. We have an special class for that: `HpSearchForTokenClassification`.
A default call to the hyperparameter search can be done with the following command.

**IMPORTANT NOTE** The `--masked_data_collator` option must be disabled to be able to perform the hyperparameter search. It is on our list to include this option. However, it is not implemented as of today.

**IMPORTANT NOTE** The `--smoke_test` option must be disabled for real training. This option generates a very fast run of `HpSearchForTokenClassification` for debugging and development purposes. However, it will not generate any good results whatsoever.

```bash

    # To test the training. It takes about 5 minutes to run
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained "EMBO/bio-lm" \
        --disable_tqdm False \
        --hyperparameter_search \
        --smoke_test 

    # For real fine tuning
    python -m smtag.cli.tokcl.train \
        --loader_path "EMBO/sd-nlp-non-tokenized" \
        --task NER \
        --from_pretrained "EMBO/bio-lm" \
        --disable_tqdm True \
        --hyperparameter_search \
        --hp_gpus_per_trial 1 \
        --hp_tune_samples 32

```

In the current implementation, the configuration classes needed to customize the `HpSearchForTokenClassification` can be located in `config.py`. In order to modify the default call, a search configuration, scheduler and reporter must be defined. We show below an example on how to define them and how to later define the `Config` object.

Research shows that [`PopulationBasedTraining`](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt) is the best performance parameter tuning algorithm today. This is the algorithm implemented in our code. A [`CLIReporter`](https://docs.ray.io/en/latest/tune/api_docs/reporters.html#clireporter) is needed  in order to get the information of the parameter finetuning process.

```python
    from ray.tune.schedulers import PopulationBasedTraining, pbt
    from ray import tune
    from ray.tune import CLIReporter

    HP_SEARCH_CONFIG = {
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32]),
            "per_device_eval_batch_size": 64,
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
            # "lr_scheduler": "cosine",
            # "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        }

    HP_SEARCH_SCHEDULER = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="eval_f1",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations={
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.loguniform(5e-4, 1e-6),
                "per_device_train_batch_size": [4, 8, 16, 32],
            },
        )

    HP_SEARCH_REPORTER = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_accuracy_score", "eval_precision", "eval_recall", "eval_f1", "epoch", "eval_runtime"],
    )

    config = Config(
        max_length=512,  # in tokens! # sentence-level: 64, abstracts/full fig captions 512 tokens
        from_pretrained="roberta-base",  # leave empty if training a language model from scratch
        model_type="Autoencoder",
        asynchr=True,  # we need ordered examples while async returns results in non deterministic way
        hp_search_config=HP_SEARCH_CONFIG,
        hp_search_scheduler=HP_SEARCH_SCHEDULER,
        hp_search_reporter=HP_SEARCH_REPORTER
    )
```
