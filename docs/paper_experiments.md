# Presenting the SoDa dataset

This document contains the commands or scripts run to generate the experiments
shown and mentioned in Abreu-Vicente & Lemberger 2022 (in prep). We take this approach
based on a strong believe on open science being the key to scientific progress.

## Panelization

From the base repository folder just call:

```bash
    sh bash_scripts/run_panelization.sh
```

The script will run through the 13 models of the paper, all except the `EMBO/bio-lm` model trained
on SMALL parts of text. The output of the process will be stored in text files in the folder
`$REPO_BASE/data/results/panelization/`. Note that if this folder does not exist, the 
script will complain about it. Just make sure you create the folder with:

```bash
    mkdir $REPO_BASE/data/results/panelization/
```
