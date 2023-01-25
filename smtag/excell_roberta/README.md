# EMBO eXcellence in life sciences RoBERTa language model

This module contains the implementation of the 
EXcell-RoBERTa language model presented in Abreu-Vicente & Lemberger 2023b (in prep). 
The pre-training weights of this model will be available in HuggingFace, as well
as the datasets used.

In order to use this model, the user should first install the SODA-RoBERTa
package as explained in the main README file of this repository.

The model can then be generated in two easy steps. The implementation of this code
defaults to the parameters presented in the paper

## Step 1: Tokenization

This is the first step to generate a language model from scratch. The tokenization
of EXcell-RoBERTa is a BPE tokenizer, similar to RoBERTa, but with the addition of
a NFKC normalizing step to normalize unicode characters. 

This is specially useful for greek letters. The tokenizer can be generated with the 
following command, given that a text file with the text corpus exists and is available.

```bash
    python -m smtag.excell_roberta.create_tokenizer \
        excell-roberta-tokenizer \
        --file_name /app/data/text/oapmc_abstracts_figs/train.txt \
        --file_name /app/data/text/oapmc_abstracts_figs/eval.txt \
        --file_name /app/data/text/oapmc_abstracts_figs/test.txt \
        --vocab_size 64000 \
        --min_freq 50
```

## Step 2: Data generation

We provide in the code the parameters used and reported in 
Abreu-Vicente & Lemberger 2023b (in prep).

Any other combination of parameters will need to be modified by the
user.

This step is included on the script to generate the model. It accepts `jsonl` files with `['text']`
as field containing the strings for input.

```
python -m smtag.excell_roberta.generate_training_data     /app/excell-roberta-tokenizer/     /app/data/json/excell_roberta/     /app/data/json/excell_roberta_training/     --block_size 510
```

## Step 3: Model pre-training

The first training can be done using:

```shell
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta/ \
    --save_steps 50000 \
    --swiglu \
    --block_size 510 \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training"
```

Successive trainings can be done from given checkpoints of the model using:

```shell
    python -m smtag.excell_roberta.model \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/excell_roberta/ \
        --loglevel info \
        --output_dir "excell-roberta-lm" \
        --resume_from_checkpoint "excell-roberta-lm"
```

Any training hyperparameters accepted by a 🤗 `TrainingArguments` class
can be parsed as optional parameters in the bash call.

## Step 4: Model evaluation with NER in Source Data

Once the language model is pretrained, it will be automatically fine tune in the 
[EMBO/sd-nlp-non-tokenized](https://huggingface.co/datasets/EMBO/sd-nlp-non-tokenized)
dataset to see its performance on the NER task.

This will be done at the end of the training process set up in the previous two comands
and the results will be printed on screen at the end of the training.

The pretraining is done using the `smtag.train.train_tokcl.TrainTokenClassification`
class and using the same parameters used in the model benchmark comparison
of Abreu-Vicente & Lemberger 2023a (in prep).

# Step 5: Model benchmarking

To give a broader view of the performancre of EXcell-RoBERTa in other biomedical datasets
we alsp fine-tune it in typical datasets used for benchmarking similar models.
This can be done using the following command:

## Training protocoll

### 1. Creation of the tokenizer

We first created the tokenizer using the following command.

```bash
    python -m smtag.excell_roberta.create_tokenizer \
        excell-roberta-tokenizer \
        --file_name /app/data/text/oapmc_abstracts_figs/train.txt \
        --file_name /app/data/text/oapmc_abstracts_figs/eval.txt \
        --file_name /app/data/text/oapmc_abstracts_figs/test.txt \
        --vocab_size 64000 \
        --min_freq 50
```

### 2. First smoke tests

A series of smoke tests were done in a small version of the train and evaluation datasets.

```shell
    python -m smtag.excell_roberta.train \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        --save_steps 50 \
        --swiglu \
        --activation gelu \
        --loglevel info \
        --output_dir "excell-roberta-training"
```

### 3. Generate the training dataset

To avoid tokenizing the dataset every single time we train the model, we 
save the tokenize data in two files.

```shell
    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/excell_roberta/ \
        /app/data/json/excell_roberta_training/ \
        --block_size 510 
```

### 3. First actual training

[22.11.2022 - 15:45]
This is the first serious training sent. We will play a bit to find a good mix of parameters

```shell
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --block_size 510 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 1000 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 0.01 \
    --logging_steps 50 \
    --max_steps 10000
```

[22.11.2022 - 16:30]
There are 22GB of data to be read in the training file. This makes the data loading difficult. We should generate 
another tokenized data versionized from the smoke test to run small tests from time to time.

```shell
    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        /app/data/json/smoke_test_training/ \
        --block_size 128 
```

[22.11.2022 - 16:40]
There was an out of memory issue in the first proper running. We run now the
train again in the smoke test to find a suitable batch size and gradient 
accumulation step combination.

[22.11.2022 - 16:40] 

The script below was run with gradient acumulation steps 2 and still out of memory happened.
The same with 4. 

```shell
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --block_size 510 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 1000 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --max_steps 10000
```
[22.11.2022 - 16:40] 

We decrease the batch size to 8 searching for a posibility to train the model.

```shell
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/smoke_text/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --block_size 510 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 1000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --max_steps 10000
```

Some issue is happening on the smoke test. We train again with the full model, but doing only 
some steps.

```shell
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/smoke_test_training/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 1000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --max_steps 150
```

Checking speed of the training comparing SwiGLU and GELU activations:
Using SwiGLU with batch size of 8 and gradient accumulation of 1 step, we get a 1.24 iterartions per second speed. 
Using GELU with batch size of 8 and gradient accumulation of 1 step, we get a 1.44 iterartions per second speed. 

This shows that SwiGLU would effectively be slower, although it seems to have also a better loss
minimizartion rate.

*I might want to test the effect of text size on the training. It seems to be a clear difference when using 512*
*token length blocks of text. I should check with 256.*
*It is confirmed. It basically goes linearly with the size. We need to think what could be the best option here.*

[23.11.2022 08:43]

While figuring out how to deal with the block size, we came back with a paper 
explaining [curriculum learning](https://aclanthology.org/2021.ranlp-1.112.pdf).
The paper shows that using a multi step aprproach is the best. 
Begin with smaller chunks of text and training them with the maximum possible
batch size. Then, repeat for consistently larger text size and smaller batch size.
This paper claims 16% less training time needed and 40% smaller loss.

We then proceed to generate the needed datasets.

```shell
    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/excell_roberta/ \
        /app/data/json/excell_roberta_training/block_size_64/ \
        --block_size 64 

    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/excell_roberta/ \
        /app/data/json/excell_roberta_training/block_size_128/ \
        --block_size 128 


    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/excell_roberta/ \
        /app/data/json/excell_roberta_training/block_size_256/ \
        --block_size 256 

    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        /app/data/json/smoke_test_training/block_size_64/ \
        --block_size 64 

    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        /app/data/json/smoke_test_training/block_size_128/ \
        --block_size 128 


    python -m smtag.excell_roberta.generate_training_data \
        /app/excell-roberta-tokenizer/ \
        /app/data/json/smoke_text/ \
        /app/data/json/smoke_test_training/block_size_256/ \
        --block_size 256 

```


[23.11.2022 13:54]

The SwiGLU activation adds a huge amount of parameters: 135 for GELU, 191 for SwiGLU. We will check if
smaller versions of SwiGLU might work. We now check the training on the smoke set for a text
block size of 64, how the GELU SwiGLU original and SwiGLU/4 work:

We will use the 64 token blocksize for the smoke text. Therefore this will also
help us to find the ideal batch size for the smallest block size.

We use this part also to find the ideal batch size for the 64 token block size

The maximum batch size possible to be used with the block size of 64 tokens is 128. It is a total of 
512 accumulating the entire distribution and parallelization.

First, Swiglu/4. The model has 149 million parameters and it generates about 1.5 it/sec

```
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/smoke_test_training/block_size_64/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --swiglu_reduction 8 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 50 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --wandb_name "excell-roberta-lm_test_swiglu_4th_param"

{'train_runtime': 342.2785, 'train_samples_per_second': 400.989, 'train_steps_per_second': 0.876, 'train_loss': 7.536332728068034, 'epoch': 50.0}
```

Then GELU
```
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/smoke_test_training/block_size_64/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 50 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --wandb_name "excell-roberta-lm_test_gelu"

{'train_runtime': 326.102, 'train_samples_per_second': 420.881, 'train_steps_per_second': 0.92, 'train_loss': 7.504174372355143, 'epoch': 50.0}
```

Then Swiglu

```
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/smoke_test_training/block_size_64/ \
    --max_length 510 \
    --init_weights 0.02 \
    --save_steps 50000 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --output_dir "excell-roberta-training" \
    --eval_steps 50 \
    --evaluation_strategy "steps" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --wandb_name "excell-roberta-lm_test_swiglu_4th_param"

{'train_runtime': 382.0929, 'train_samples_per_second': 359.206, 'train_steps_per_second': 0.785, 'train_loss': 7.433011983235677, 'epoch': 50.0}
```

[23.11.2022 15:11] Checking the best batch size for the other models.

Using GELU as intermediate activation function these are the results obtained.

| Block size | Batch size |
|------------|------------|
| 64         | 128        |
| 128        | 64         |
| 256        | 32         |
| 512        | 8          |

[24.11.2022 12:00] Checking the amount of resources needed for training in DGX.
We can run knowledge graph generations in the meantime.

[25.11.2022 09:00]

Sending a day long training using SwiGLU and GELU to find which of the two options is the best.
This is done for 64, 128 and 512 block size. This experiment shows that SwiGLU activations will generate a training that is 4h longer per epoch.
We need to check what the improvement of the SwiGLU activation is to see whether it even makes any sense.

| Block size | Batch size | Activation | Total steps | Time expected | Total examples | Total tokens | Training epoch | Training loss | Eval loss |
|------------|------------|------------|------------:|---------------|---------------:|-------------:|----------------|---------------|-----------|
|     64     |     128    | SwiGLU     |       32.8K |      12h      |          33.4M |        2.15B | 0.5            |               |           |
|     64     |     128    | GELU       |       32.8K |      10h      |          33.4M |        2.15B | 0.5            |               |           |
|     128    |     64     | SwiGLU     |       32.8K |    12.25h     |          16.7M |        2.15B | 0.5            |               |           |
|     128    |     64     | GELU       |       32.8K |    10.30h     |          16.7M |        2.15B | 0.5            |               |           |
|     256    |     32     | SwiGLU     |       32.8K |               |           8.4M |        2.15B | 0.5            |               |           |
|     256    |     32     | GELU       |       32.8K |               |           8.4M |        2.15B | 0.5            |               |           |
|     512    |      8     | SwiGLU     |       65.6K |     13.5h     |           4.2M |        2.15B | 0.5            |               |           |
|     512    |      8     | GELU       |       65.6K |     11.5h     |           4.2M |        2.15B | 0.5            |               |           |

```bsh
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_64/ \
    --per_device_train_batch_size 128 \
    --learning_rate 0.0005 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block64-swiglu/" \
    --wandb_name "test-block64-swiglu"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_64/ \
    --per_device_train_batch_size 128 \
    --learning_rate 0.0005 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block64-gelu/" \
    --wandb_name "test-block64-gelu"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --per_device_train_batch_size 8 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.025 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block512-swiglu/" \
    --wandb_name "test-block512-swiglu"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --per_device_train_batch_size 8 \
    --learning_rate 0.0001 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block512-gelu/" \
    --wandb_name "test-block512-gelu"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_128/ \
    --per_device_train_batch_size 64 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block128-swiglu/" \
    --wandb_name "test-block128-swiglu"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_128/ \
    --per_device_train_batch_size 64 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 32000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/test-block128-gelu/" \
    --wandb_name "test-block128-gelu"
```

[28.11.2022 - 10AM]

The results of the previous experiment show very clear that
SwiGLU and curriculum are the best options to run the training. 
We therefore will use both of them.

We begin now in the checkpoint of block size 64 and 5000 steps with SwiGLU and
]resume the training for another 10000 steps using SwiGLU and
128 blocksize.

```bsh
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_128/ \
    --resume_from_checkpoint /app/soda-roberta/lm_models/test-block128-swiglu/checkpoint-6000 \
    --per_device_train_batch_size 64 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 40 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.0 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/excell-roberta/lm_models/cv-training-phase-128/" \
    --wandb_name "cv-training-phase-128"

```

The result has not been so convincing.

[28.11.2022 - 3PM]

We do a test similar to the previous but we will short th steps to be trained before jumping to a different block size.

```bsh
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_64/ \
    --per_device_train_batch_size 128 \
    --learning_rate 0.0005 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --max_steps 2001 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --logging_first_step \
    --output_dir "/app/excell-robetra/lm_models/training-step-64" \
    --wandb_name "training-step-64"

python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_128/ \
    --resume_from_checkpoint /app/excell-robetra/lm_models/training-step-64/checkpoint-2000/ \
    --per_device_train_batch_size 64 \
    --learning_rate 0.0005 \
    --lr_scheduler_type "linear" \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --eval_steps 3001 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.5 \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --logging_first_step \
    --output_dir "/app/excell-robetra/lm_models/training-step-128" \
    --wandb_name "training-step-128"


python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --resume_from_checkpoint /app/excell-robetra/lm_models/training-step-128/checkpoint-3000/ \
    --per_device_train_batch_size 8 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 5000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1. \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --logging_first_step \
    --output_dir "/app/soda-roberta/lm_models/training-step-512/" \
    --wandb_name "training-step-512"
```


Not reallly improved anything. We keep the normal training on
the largest block size

[28.11.2022 - 10PM]

```bsh
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --resume_from_checkpoint soda-roberta/lm_models/test-block512-swiglu/checkpoint-20000 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "linear" \
    --warmup_steps 1000 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 5000 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.0 \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --logging_first_step \
    --output_dir "/app/lm_models/excell-roberta/bs-512-training-2/" \
    --wandb_name "bs-512-training-2"
```

[01.12.2022 11AM] The first epoch training has now successfully finished. 
We run a NER finetuning on the SourceData dataset.

```bsh

python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --model "lm_models/excell-roberta/bs-512-training-2/checkpoint-120000" \
    --add_prefix_space

                precision    recall  f1-score   support

          CELL       0.67      0.80      0.73      2199
     EXP_ASSAY       0.55      0.57      0.56      4092
      GENEPROD       0.72      0.89      0.79      6513
      ORGANISM       0.66      0.81      0.73      1001
SMALL_MOLECULE       0.64      0.73      0.68      1319
   SUBCELLULAR       0.67      0.59      0.63       881
        TISSUE       0.64      0.77      0.70      1054

     micro avg       0.66      0.76      0.71     17059
     macro avg       0.65      0.74      0.69     17059
  weighted avg       0.65      0.76      0.70     17059

```

The results are already on the order of magnitude of other models. 
We will now first check the CRF implementation and then check the fine tuning with it. 

The next step will be to train the model as much as another series of epochs to 
see how this is evolving.

```bsh
python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --model "lm_models/excell-roberta/bs-512-training-2/checkpoint-120000" \
    --add_prefix_space \
    --crf
                precision    recall  f1-score   support                                                                                                                                                             
                                                                                                                                                                                                                    
          CELL       0.57      0.63      0.60      2199                                                                                                                                                             
     EXP_ASSAY       0.52      0.48      0.50      4092                                                                                                                                                             
      GENEPROD       0.59      0.77      0.67      6513                                                                                                                                                             
      ORGANISM       0.61      0.70      0.66      1001                                                                                                                                                             
SMALL_MOLECULE       0.57      0.46      0.51      1319                                                                                                                                                             
   SUBCELLULAR       0.64      0.46      0.53       881                                                                                                                                                             
        TISSUE       0.57      0.63      0.60      1054                                                                                                                                                             
                                                                                                                                                                                                                    
     micro avg       0.57      0.63      0.60     17059                                                                                                                                                             
     macro avg       0.58      0.59      0.58     17059                                                                                                                                                             
  weighted avg       0.57      0.63      0.59     17059                                                                                                                                                             

```

[01.12.2022 2PM] It looks like CRF is not working better than the single head.
We send now a training for another set of epochs.

```bsh
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --resume_from_checkpoint lm_models/excell-roberta/checkpoint-1-epoch/ \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "linear" \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 5000 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4.0 \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --logging_first_step \
    --output_dir "/app/lm_models/excell-roberta/training-step-3/" \
    --wandb_name "training-step-3"
```


```
python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --model "lm_models/excell-roberta/training-step-3/checkpoint-520000/" \
    --add_prefix_space 

```

**We obtained a 0.74 in the best F score. But found a bug!!!!**
The option `add_prefix_space` was set to false when the tokenizer was created.
This is a problem since it needs to be set to True. We will need to repeat now the entire training
process.

On this training process we will do a series of differences.

1. We add extra tokens to the data, coming from full papers. Goal is to hit the 4Billion token mark
2. We train the first epoch with 128 block size
3. We resume the rest of the training with 512 block size
4. We will train for about 20 epochs or until divergence of the evaluation metrics

# 1. New correct tokenization

We added papers to the dataset. The idea is that the dataset contains about 4B tokens, which should be the ideal 
amount of training tokens for the size of the model we are training.

```bash
python -m smtag.excell_roberta.create_tokenizer \
    excell-roberta-tokenizer \
    --file_name /app/data/text/oapmc_abstracts_figs/train.txt \
    --file_name /app/data/text/oapmc_abstracts_figs/eval.txt \
    --file_name /app/data/text/oapmc_abstracts_figs/test.txt \
    --vocab_size 64000 \
    --min_freq 50

Roberta -Excell-roberta     PMB
['<s>', 'card', 'i', 'omy', 'ocyte', '</s>'] ['<s>', 'cardi', '##omy', '##ocyte', '</s>'] ['[CLS]', 'cardiomyocyte', '[SEP]']
['<s>', 'di', 'abetes', '</s>'] ['<s>', 'diabetes', '</s>'] ['[CLS]', 'diabetes', '[SEP]']
['<s>', 'le', 'ukemia', '</s>'] ['<s>', 'leuk', '##emia', '</s>'] ['[CLS]', 'leukemia', '[SEP]']
['<s>', 'l', 'ith', 'ium', '</s>'] ['<s>', 'l', '##ith', '##ium', '</s>'] ['[CLS]', 'lithium', '[SEP]']
['<s>', 'ins', 'ulin', '</s>'] ['<s>', 'insulin', '</s>'] ['[CLS]', 'insulin', '[SEP]']
['<s>', 'DNA', '</s>'] ['<s>', 'DNA', '</s>'] ['[CLS]', 'dna', '[SEP]']
['<s>', 'd', 'na', '</s>'] ['<s>', 'd', '##na', '</s>'] ['[CLS]', 'dna', '[SEP]']
['<s>', 'prom', 'oter', '</s>'] ['<s>', 'promoter', '</s>'] ['[CLS]', 'promoter', '[SEP]']
['<s>', 'hy', 'pert', 'ension', '</s>'] ['<s>', 'hypertension', '</s>'] ['[CLS]', 'hypertension', '[SEP]']
['<s>', 'n', 'eph', 'rop', 'athy', '</s>'] ['<s>', 'ne', '##ph', '##ropathy', '</s>'] ['[CLS]', 'nephropathy', '[SEP]']
['<s>', 'ly', 'mph', 'oma', '</s>'] ['<s>', 'lymphoma', '</s>'] ['[CLS]', 'lymphoma', '[SEP]']
['<s>', 'l', 'id', 'oc', 'aine', '</s>'] ['<s>', 'l', '##id', '##ocaine', '</s>'] ['[CLS]', 'lidocaine', '[SEP]']
['<s>', 'or', 'oph', 'aryn', 'ge', 'al', '</s>'] ['<s>', 'o', '##roph', '##aryngeal', '</s>'] ['[CLS]', 'oropharyngeal', '[SEP]']
['<s>', 'chlor', 'amp', 'hen', 'icol', '</s>'] ['<s>', 'chlor', '##amphenicol', '</s>'] ['[CLS]', 'chloramphenicol', '[SEP]']
['<s>', 'Rec', 'A', '</s>'] ['<s>', 'Rec', '##A', '</s>'] ['[CLS]', 'reca', '[SEP]']
['<s>', 're', 'ca', '</s>'] ['<s>', 'rec', '##a', '</s>'] ['[CLS]', 'reca', '[SEP]']
['<s>', 'acet', 'yl', 'transfer', 'ase', '</s>'] ['<s>', 'acetyl', '##transferase', '</s>'] ['[CLS]', 'acetyltransferase', '[SEP]']
['<s>', 'cl', 'on', 'idine', '</s>'] ['<s>', 'cl', '##on', '##idine', '</s>'] ['[CLS]', 'clonidine', '[SEP]']
['<s>', 'n', 'al', 'ox', 'one', '</s>'] ['<s>', 'n', '##al', '##oxone', '</s>'] ['[CLS]', 'naloxone', '[SEP]']


```

# 2. Generate the next training datasets


```
python -m smtag.excell_roberta.generate_training_data \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta/ \
    /app/data/json/excell_roberta_training/block_size_512/ \
    --block_size 510

l   
```
# 3. Train the first epoch on blocksize of 128 tokens

```
python -m smtag.excell_roberta.train \
    /app/excell-roberta-tokenizer/ \
    /app/data/json/excell_roberta_training/block_size_128/ \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --max_steps 50001 \
    --learning_rate 0.00025 \
    --lr_scheduler_type "linear" \
    --warmup_steps 1000 \
    --max_length 510 \
    --init_weights 0.02 \
    --swiglu \
    --swiglu_reduction 2 \
    --whole_word_masking \
    --activation gelu \
    --loglevel info \
    --save_strategy "steps" \
    --save_steps 10000 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --logging_first_step \
    --output_dir "/app/excell-roberta/lm_models/v3-training-step-128" \
    --wandb_name "v3-training-v2-epoch-1-128bs"
```

Doing a checkup on the fine-tuning task:

``` bash
python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task NER \
    --model "/app/excell-robetra/lm_models/v2-training-step-128/checkpoint-106556" \
    --add_prefix_space 
```

# 4. Set of benchmarking for comparison with PubMedBERT

## SourceData dataset

## Entire dataset

PubMedBERT - 77, BioLinkBERT - 76, SoDa-lm - 75, BioMegatron - 74

### Fully connected head

                precision    recall  f1-score   support
                                                                                                                                                                                                                                                                                                                                                                          
          CELL       0.83      0.86      0.84      2202 
       DISEASE       0.57      0.60      0.58       197
     EXP_ASSAY       0.65      0.72      0.69      4091  
      GENEPROD       0.90      0.93      0.91      6520 
      ORGANISM       0.80      0.89      0.84      1003
SMALL_MOLECULE       0.76      0.85      0.80      1320
   SUBCELLULAR       0.75      0.74      0.74       883
        TISSUE       0.75      0.78      0.76      1030
        
     micro avg       0.79      0.84      0.82     17246
     macro avg       0.75      0.80      0.77     17246
  weighted avg       0.79      0.84      0.82     17246


### Fully connected and CRF head


                precision    recall  f1-score   support
                                                                                                                                                                                                                                                                                                                                                                          
          CELL       0.84      0.86      0.85      2202
       DISEASE       0.62      0.59      0.61       197
     EXP_ASSAY       0.66      0.73      0.69      4091
      GENEPROD       0.90      0.93      0.92      6520
      ORGANISM       0.81      0.89      0.85      1003
SMALL_MOLECULE       0.77      0.84      0.81      1320
   SUBCELLULAR       0.78      0.75      0.77       883
        TISSUE       0.76      0.79      0.78      1030 
        
     micro avg       0.80      0.84      0.82     17246
     macro avg       0.77      0.80      0.78     17246
  weighted avg       0.80      0.84      0.82     17246

### Fully connected and CRF head and include start end transitions

                precision    recall  f1-score   support

```          CELL       0.83      0.86      0.84      2202``` 
```       DISEASE       0.63      0.60      0.62       197``` 
```     EXP_ASSAY       0.66      0.72      0.69      4091```
```      GENEPROD       0.91      0.93      0.92      6520```
      ORGANISM       0.83      0.90      0.86      1003
SMALL_MOLECULE       0.78      0.84      0.81      1320
   SUBCELLULAR       0.78      0.74      0.76       883
        TISSUE       0.77      0.79      0.78      1030

     micro avg       0.80      0.84      0.82     17246
     macro avg       0.77      0.80      0.78     17246
  weighted avg       0.81      0.84      0.82     17246

### PubMedBERT comparison

                precision    recall  f1-score   support
                                                                                                                                                                                                                                                                                                                                                                          
          CELL       0.81      0.84      0.82      4948
       DISEASE       0.49      0.47      0.48       463
     EXP_ASSAY       0.67      0.68      0.68      9885
      GENEPROD       0.90      0.93      0.91     21865
```      ORGANISM       0.84      0.89      0.87      3464```
```SMALL_MOLECULE       0.83      0.85      0.84      6431```
```   SUBCELLULAR       0.80      0.78      0.79      3850```
```        TISSUE       0.76      0.78      0.77      2975```

     micro avg       0.82      0.84      0.83     53881
     macro avg       0.76      0.78      0.77     53881
  weighted avg       0.82      0.84      0.83     53881

### Biolink bert large
                precision    recall  f1-score   support

          CELL       0.83      0.84      0.83      4948
       DISEASE       0.46      0.66      0.54       463
     EXP_ASSAY       0.66      0.69      0.68      9885
      GENEPROD       0.91      0.94      0.92     21865
      ORGANISM       0.83      0.91      0.87      3464
SMALL_MOLECULE       0.83      0.87      0.85      6431
   SUBCELLULAR       0.78      0.81      0.80      3850
        TISSUE       0.76      0.81      0.78      2975

     micro avg       0.82      0.86      0.84     53881
     macro avg       0.76      0.82      0.78     53881
  weighted avg       0.82      0.86      0.84     53881

## BC2GM

PubMedBERT - 84

### Fully connected head

              precision    recall  f1-score   support

        GENE       0.76      0.81      0.78      3659

   micro avg       0.76      0.81      0.78      3659
   macro avg       0.76      0.81      0.78      3659
weighted avg       0.76      0.81      0.78      3659

### Fully connected and CRF head

              precision    recall  f1-score   support

        GENE       0.80      0.81      0.81      3659

   micro avg       0.80      0.81      0.81      3659
   macro avg       0.80      0.81      0.81      3659
weighted avg       0.80      0.81      0.81      3659

## BC5CDR Chem

BioLinkBERT (large) - 94

### Fully connected and CRF head

              precision    recall  f1-score   support

    Chemical       0.91      0.95      0.93      3559

   micro avg       0.91      0.95      0.93      3559
   macro avg       0.91      0.95      0.93      3559
weighted avg       0.91      0.95      0.93      3559

## BC5CDR Disease
BioLinkBERT (large) - 86.4, BioMegatron (88.5)

### Fully connected head

              precision    recall  f1-score   support
                                                                                                                                                                                                                    
     Disease       0.75      0.82      0.79      2861 
   micro avg       0.75      0.82      0.79      2861
   macro avg       0.75      0.82      0.79      2861
weighted avg       0.75      0.82      0.79      2861

### Fully connected and CRF head

              precision    recall  f1-score   support

     Disease       0.74      0.84      0.79      2861

   micro avg       0.74      0.84      0.79      2861
   macro avg       0.74      0.84      0.79      2861
weighted avg       0.74      0.84      0.79      2861

## JNLPBA

BioLinkBERT (large) - 80.06, PubMedBERT - 79.1

### Fully connected head


### Fully connected and CRF head


## NCBI Disease
BioBERT - 89.71


### Fully connected and CRF head

              precision    recall  f1-score   support

     Disease       0.81      0.90      0.85       574

   micro avg       0.81      0.90      0.85       574
   macro avg       0.81      0.90      0.85       574
weighted avg       0.81      0.90      0.85       574

## Checking the generic roles with EXcell-RoBERTa

``` bash
python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task ROLES \
    --model "/lm_models/excell-roberta/v3-training-512bs/checkpoint-2716290" \
    --add_prefix_space \
    --num_train_epochs 2.


                precision    recall  f1-score   support

CONTROLLED_VAR       0.76      0.82      0.79      1952
  MEASURED_VAR       0.85      0.93      0.89     23440

     micro avg       0.84      0.92      0.88     25392
     macro avg       0.80      0.87      0.84     25392
  weighted avg       0.84      0.92      0.88     25392

```

## Checking the panelization with EXcell-RoBERTa

``` bash
python -m smtag.excell_roberta.token_classification \
    --data "EMBO/sd-nlp-non-tokenized" \
    --task PANELIZATION \
    --model "/lm_models/excell-roberta/v3-training-512bs/checkpoint-2716290" \
    --add_prefix_space \
    --num_train_epochs 1. \
    --run_name "sd-panelization-v2" \
    --push_to_hub \
    --hub_model_id "EMBO/sd-panelization-v2" \
    --hub_strategy "end" \
    --hub_token ""

                  precision    recall  f1-score   support

 PANEL_START       0.97      0.99      0.98      1307

   micro avg       0.97      0.99      0.98      1307
   macro avg       0.97      0.99      0.98      1307
weighted avg       0.97      0.99      0.98      1307

```


# Entire change on the process due to tokenization issues

```
python -m smtag.excell_roberta.token_classification "/data/xml/sd_panels_filtered/" \
    --task NER \
    --tokenizer "/app/excell-roberta-tokenizer/" \
    --model "/lm_models/excell-roberta/v3-training-512bs/checkpoint-2716290" \
    --max_length 500 \
    --num_train_epochs 1.0
```

```
python -m smtag.excell_roberta.token_classification "/data/xml/sd_panelization_filtered/" \
    --task PANELIZATION \
    --tokenizer "/app/excell-roberta-tokenizer/" \
    --model "/lm_models/excell-roberta/v3-training-512bs/checkpoint-2716290" \
    --max_length 512
```