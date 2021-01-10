
SODA-ROBERTA
============

SODA-ROBERTA is a **So**urce **Da**ta reource for training __Roberta__ transfomers for nlp tasks in cell and molecular biology.
SourceData database: https://sourcedata.io, "SourceData: a semantic platform for curating and searching figures"
Liechti R, George N, Götz L, El-Gebali S, Chasapi A, Crespo I, Xenarios I, Lemberger T, Nature Methods, https://doi.org/10.1038/nmeth.4471
Roberta transformers is a BERT derivative: https://huggingface.co/transformers/model_doc/roberta.html, "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

The resource uses the huggingface (https://huggingface.co) and PyTorch frameworks.


Create the appropriate layout

```bash
mkdir ner_dataset
mkdir tokcl_models  # NER_MODEL_PATH
mkdir cache  # HUGGINGFACE_CACHE
```

Edit .env.example and save as .env

Build and start the container:

```bash
docker-compose build
docker-compose up -d
```
Start a tmux session and run bash from the lm service:

```bash
tmux
docker-compose run --rm lm bash
```

Split the original documents into train, eval and test sets. This is done at the document level since each document may contain several examples. Doing the split already now ensures more independent eval and test sets.

```bash
python -m common.split 191012/ -X xml
```

Extract the examples for NER using an XPAth that identifies individual panel legends within figure legends:

```bash
mkdir sourcedata
python -m common.extract 191012/train sourcedata/train -P .//sd-panel --keep-xml
python -m common.extract 191012/eval sourcedata/eval -P .//sd-panel --keep-xml
python -m common.extract 191012/test sourcedata/test -P .//sd-panel --keep-xml
```

Same thing but using a XPath for entire figure legends encompassing several panel legends. This will be used to learn segmentation of figure legends into panel legends:

```bash
mkdir panelization
python -m common.extract 191012/train panelization/train -P .//fig --keep-xml
python -m common.extract 191012/eval panelization/eval -P .//fig --keep-xml
python -m common.extract 191012/test panelization/test -P .//fig --keep-xml
```

Prepare the dataset for NER and ROLE labels:

```bash
python -m tokcl.dataprep sourcedata
```

Train the NER task to learn entity types:

```bash
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
--save_steps=100 \
```


Train the ROLES task to learn entity roles:

```bash
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

```bash
rm -fr ner_dataset  # dataprep does not overwrite to avoid disasters
python -m tokcl.dataprep panelization
```

Train the PANELIZATION task to learn panel segmentation:

```bash
python -m tokcl.train PANELIZATION \
--output_dir=ner_model/PANELIZATION \ \
--overwrite_output_dir \ \
--learning_rate=1e-5 \ \
--num_train_epochs=100 \ \
--per_device_train_batch_size=32 \ \
--per_device_eval_batch_size=32 \ \
--evaluation_strategy='steps' \ \
--save_total_limit=3 \ \
--logging_steps=50 \ \
--eval_steps=50 \ \
--save_steps=100
```

Try smtag tagging:

```bash
python -m infer.smtag "We studied mice with genetic ablation of the ERK1 gene in brain and muscle."
```



