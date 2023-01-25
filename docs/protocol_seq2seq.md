# Analyzing Seq2Seq models as oposed to the classical NER tagging

## Finetuning

Similar to the finetuning of the token classifier. However, in this case we ask the model to give the answer as text instead of creating labels. 
We will do this using different models for the tasks NER, PANELIZATION, Causal hypothesis.

Previous research shows that RoBERTa tokenized models worked better. We therefore expect better performance from BART and BioBART than from the T5 based models.

At this point we will run the model to see the training. We still need to generate predictions and measure the performance of the models.
This will come as a second step.

### Causal hypothesis of GENEPROD

python -m smtag.cli.seq2seq.hf_finetune "/app/data/seq2seq/GENEPROD_seq2seq_cleaned.csv" CAUSAL "copy_tag" \
    --delimiter "###tt9HHSlkWoUM###" \
    --base_model "t5-base" \
    --do_train \
    --do_predict \
    --do_eval \
    --max_input_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 15. \
    --learning_rate 0.00005 \
    --lr_schedule "cosine" \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --early_stopping \
    --logging_steps 100 \
    --temperature 0.75 \
    --run_name "seq2seq-t5-causal" \
    --generation_max_length 512 \
    --predict_with_generate 

python -m smtag.cli.seq2seq.hf_finetune "/app/data/seq2seq/GENEPROD_seq2seq_cleaned.csv" CAUSAL "copy_tag" \
    --delimiter "###tt9HHSlkWoUM###" \
    --base_model "facebook/bart-base" \
    --do_train \
    --do_predict \
    --do_eval \
    --max_input_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 15. \
    --learning_rate 0.00005 \
    --lr_schedule "cosine" \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --early_stopping \
    --logging_steps 100 \
    --temperature 0.75 \
    --run_name "seq2seq-bart-causal" \
    --generation_max_length 512 \
    --predict_with_generate 

### NER

#### facebook/bart-base

#### facebook/bart-large

#### facebook/t5-base

python -m smtag.cli.seq2seq.hf_finetune "EMBO/sd-nlp-non-tokenized" "NER" "copy_tag" \
         --base_model "t5-base" \
         --ner_labels GENEPROD EXP_ASSAY SMALL_MOL \
         --generate_end "[END]" \
         --prompt_init "Copy the input and label the GENEPROD, SMALL_MOL and EXP_ASSAY entities: " \
         --max_input_length 512 \
         --max_target_length 512 \
         --per_device_train_batch_size 4 \
         --per_device_eval_batch_size 4 \
         --num_train_epochs 10. \
         --learning_rate 0.00005 \
         --lr_schedule "cosine" \
         --evaluation_strategy "steps" \
         --eval_steps 1000 \
         --save_steps 2000 \
         --save_total_limit 5 \
         --early_stopping \
         --do_train \
         --do_eval \
         --do_predict \
         --logging_steps 100 \
         --temperature 0.75 \
         --run_name "seq2seq-biobart-base-ner-all" \
         --generation_max_length 512 \
         --predict_with_generate \
         --generation_num_beams 1 \
         --no_repeat_ngram_size 3

#### razent/SciFive-base-Pubmed

#### GanjinZero/biobart-base

#### GanjinZero/biobart-large

```bsh                                                                             

python -m smtag.cli.seq2seq.hf_finetune "EMBO/sd-nlp-non-tokenized" "NER" "copy_tag" \
         --base_model "t5-base" \
         --ner_labels GENEPROD EXP_ASSAY SMALL_MOL \
         --generate_end "[END]" \
         --prompt_init "Copy the input and label the GENEPROD, SMALL_MOL and EXP_ASSAY entities: " \
         --max_input_length 512 \
         --max_target_length 512 \
         --per_device_train_batch_size 4 \
         --per_device_eval_batch_size 4 \
         --num_train_epochs 10. \
         --learning_rate 0.00005 \
         --lr_schedule "cosine" \
         --evaluation_strategy "steps" \
         --eval_steps 1000 \
         --save_steps 2000 \
         --save_total_limit 5 \
         --early_stopping \
         --do_train \
         --do_eval \
         --do_predict \
         --logging_steps 100 \
         --temperature 0.75 \
         --run_name "seq2seq-biobart-base-ner-all" \
         --generation_max_length 512 \
         --predict_with_generate \
         --generation_num_beams 1 \
         --no_repeat_ngram_size 3


python -m smtag.cli.seq2seq.hf_finetune "EMBO/sd-nlp-non-tokenized" "NER" \
         --base_model "GanjinZero/biobart-large" \
         --from_local_checkpoint "./test_seq2seq_metrics/checkpoint-30000/" \
         --ner_labels all \
         --generate_end "[END]" \
         --max_input_length 512 \
         --max_target_length 64 \
         --per_device_train_batch_size 8 \
         --per_device_eval_batch_size 8 \
         --early_stopping \
         --do_predict \
         --logging_steps 100 \
         --run_name "seq2seq-biobart-large-ner-all" \
         --generation_max_length 64 \
         --predict_with_generate \
         --generation_num_beams 1


```


### PANELIZATION

It does not work so good on the smaller models. Few shot learning works much better with just 2 or 3 examples.
Example of a call to it.

```bsh
    python -m smtag.cli.seq2seq.hf_finetune "/data/seq2seq/panelization_task.csv" "other" "other" \
        --base_model "GanjinZero/biobart-base" \
        --max_input_length 1024 \
        --max_target_length 1024 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --num_train_epochs 50. \
        --learning_rate 0.0001 \
        --evaluation_strategy "steps" \
        --eval_steps 1000 \
        --save_total_limit 10 \
        --do_train \
        --do_eval \
        --do_predict \
        --logging_steps 100 \
        --run_name "seq2seq-biobart-base-panelization" \
        --generation_max_length 1024 \
        --predict_with_generate \
        --generation_num_beams 1

```



## Few-shot
With few-shot, a model is giving a few example and from there it is able to generalize. This is possible
only for the largest models, such as BLOOM and GPT-3. Previous attemps seem to show that this would not be enough for
NER, but it might be good enough for panelization. 

We run experiments in both, BLOOM and GPT-3, to show this point.


## Issues with the dataset

### Bad tagging in the panelization task

This is an example of a badly tgged figure for the panelization task.

target:                                                                                                                                                                              
 <panel> (A) Representative immunofluorescence images of ZW10 streaming from KTs in Drosophila S2 cells in metaphase expressing either SpindlyWT-, SpindlyS499A- or SpindlyS499D-EGFP
. ZW10 streaming is also shown in S2 cells expressing SpindlyWT- or SpindlyS499A-EGFP in a dynein-depleted background. Insets display magnifications of the outlined regions, which h
ighlight streaming robustness. **(B) Graph represents the percentage of cells in metaphase showing different levels of ZW10 streaming (n≥35 cells for each condition, n=2 independent e
xperiments). (C) Graph represents the percentage of ZW10 levels at aligned KTs normalized to ZW10 levels at the spindle region. All values were normalized to the mean fluorescence i
ntensity quantified in SpindlyWT-EGFP expressing cells, which was set to 100% (n≥470 KTs from at least 35 cells for each condition, n=2 independent experiments). Data information: S
tatistical analysis was calculated using a Kruskal-Wallis test for multiple comparisons. p values: ****, &lt;0.0001. Data are shown as mean ± SD. Scale bar: 5 μm.** </panel>          
 <panel> (D) Representative immunofluorescence images of calcium-stable KT-MT attachments in metaphase S2 cells expressing SpindlyWT-, SpindlyS499A-, SpindlyS499D- or SpindlyS499D-E
GFP in a ZW10-depleted background. Insets display magnifications of the outlined regions which highlight different attachment configurations (E - end on; L - lateral; M - merotelic)
. Cartoon depicts the attachment configuration of the respective KT pair. Asterisk highlights an aligned KT pair in which a sister KT appears to be laterally attached to the end of 
a MT fiber. Plotted profiles show the overlap between CENP-C and tubulin signals for the highlighted KT. CENP-C was used as a KT reference. **(E) Graph represents the percentage of me
taphase cells showing only end-on attachments or at least 1 KT with lateral or merotelic attachment, as shown in D (n≥44 cells for each condition, n≥2 independent experiments). Data
 information: Statistical analysis was calculated using a Kruskal-Wallis test for multiple comparisons. p values: ****, &lt;0.0001. Data are shown as mean ± SD. Scale bar: 5 μm.** </p
anel>                                                                                                                                                                                
 <panel> (F) Proposed model for Polo-mediated regulation of RZZ-Spindly and its impact on KT-MT attachment. High levels of active Polo during early mitosis render Spindly phosphoryl
ated on Ser499, thus weakening its interaction with the RZZ complex at KTs. On laterally attached KTs, the presence of minus end-directed motor dynein allows the removal of phosphor
ylated Spindly while the RZZ complex is retained at high levels to inhibit premature formation of stable end-on attachments and therefore avoid merotely. As protein phosphatases are
 recruited to KTs, Polo activation status declines and Spindly is eventually dephosphorylated. Spindly is now able to bind Zwilch and the Spindly-RZZ complex is stripped from KTs th
rough dynein, which enables the stable conversion from lateral to end-on attachments and SAC silencing. </panel> [END]                                                               
