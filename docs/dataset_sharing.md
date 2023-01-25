# Post the datasets

Clone the EMBO/sd-nlp repo if not yet done:

    git clone https://huggingface.co/datasets/EMBO/sd-nlp
    cd sd-nlp

Copy the relevant dataset files

    cp ../soda-roberta/data/json/220304_sd_panels.zip ./sd_panels.zip
    cp ../soda-roberta/data/json/220304_sd_fig.zip ./sd_figs.zip
    

Copy the loading script:

    cp ../soda-roberta/smtag/loader/loader_tokcl.py sd-nlp.py

Make sure .jsonl files are tracked as lfs objects:

    git lfs install
    git lfs track *.zip
    git add .gitattributes
    git commit -m "tracking *.zip files"

Add and commit files:

    git status
    git add -A
    git commit -m "updating data files and loader script"
    git push

Create and update the dataset card https://huggingface.co/docs/datasets/v2.0.0/en/dataset_card
using the appropriate dataset tagging https://huggingface.co/spaces/huggingface/datasets-tagging
