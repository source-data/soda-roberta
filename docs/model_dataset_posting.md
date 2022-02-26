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

