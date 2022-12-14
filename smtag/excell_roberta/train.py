import argparse

from datasets import load_dataset, Dataset

import glob
import logging
import os

from smtag.data_classes import TrainingExcellRoberta, TrainingArgumentsTOKCL

from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleLM
from .configuration_excell_roberta import EXcellRobertaConfig
from .modeling_excell_roberta import EXcellRobertaForMaskedLM

from transformers import (DataCollatorForLanguageModeling, HfArgumentParser,
                            RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM,
                            Trainer, DataCollatorForWholeWordMask)
import wandb

logger = logging.getLogger('smtag.excell_roberta.model')



if __name__ == "__main__":

    # The first part of the code would be the argument parsing
    parser = HfArgumentParser(TrainingExcellRoberta, description="Traing script.")
    # parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="Path (absolute) or name to the tokenizer to be used to train the model")
    parser.add_argument("datapath", help="Path (absolute) tothe data that will be used to train the model")
    parser.add_argument("--max_length", default=512, type=int, help="Max length of the model")
    parser.add_argument("--from_checkpoint", default="", type=str, help="Checkpoint from which to initialize the trainingn of the model.")
    parser.add_argument("--init_weights", default=0.02, type=float, help="Range of the initial weights of the model")
    parser.add_argument("--activation", default="gelu", type=str, help="Activation function to be used")
    parser.add_argument("--swiglu", action="store_true", help="If using gelu as activation for intermediate layers")
    parser.add_argument("--swiglu_reduction", default=2, type=int, help="Reducing factor for SwiGLU to minimize parameters and speed up training")
    parser.add_argument("--whole_word_masking", action="store_true", help="Activates whole word masking")
    parser.add_argument("--wandb_name", default="excell-roberta-lm", type=str, help="Name for the wandb UI")
    parser.add_argument( '-log',
                        '--loglevel',
                        default='warning',
                        help='Provide logging level. Example --loglevel debug, default=warning' ) 
    training_args, args = parser.parse_args_into_dataclasses()
    logging.basicConfig( level=args.loglevel.upper() )

    # The second part is defining the tokenizer, configuration, and model
    wandb.init(project="excell-roberta", entity="embo", name=args.wandb_name)

    # The second part is defining the tokenizer, configuration, and model
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer, max_len=args.max_length)

    config = EXcellRobertaConfig(
        # #Common model attributes
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        position_embedding_type="absolute",
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
        hidden_act=args.activation,
        swiglu=args.swiglu,
        swiglu_reduction=2,
        initializer_range=args.init_weights,
        type_vocab_size=1,
        max_position_embeddings=514,
        intermediate_size=3072,
        bias_dense_layers=False,
        bias_norm=False,
        #Tokenizer parameters
        tokenizer_class=tokenizer.__class__.__name__,
        prefix=None,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        #Class attributes
        # model_type="roberta",
    )
    
    if training_args.resume_from_checkpoint:
        model = EXcellRobertaForMaskedLM.from_pretrained(training_args.resume_from_checkpoint)
    else:
        model = EXcellRobertaForMaskedLM(config=config)

    logger.info(100*"*")
    logger.info(f"You are creating a new model {model.__class__.__name__} from scratch")
    logger.info(f"The model has {int(model.num_parameters()/1e6)} million parameters")
    logger.info(print(model))
    logger.info(100*"*")

    # The third part of the code is generating the data
    logger.info("Reading the dataset")

    ds = load_dataset("json", data_files={'train': [os.path.join(args.datapath, "train.jsonl")],
                                        'eval': os.path.join(args.datapath, "eval.jsonl")})
        
    logger.info(print(ds))
    # The fourth part is generating a data collator for masked language modelling
    if args.whole_word_masking:
        data_collator = DataCollatorForWholeWordMask(tokenizer,
                                                            mlm=True,
                                                            mlm_probability=0.2,
                                                        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer,
                                                            mlm=True,
                                                            mlm_probability=0.2,
                                                        )
    # The next step is to generate the training, inizializing training arguments too
    logging.info("Training the new excell-roberta model")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        callbacks=[ShowExampleLM(tokenizer),
                    MyTensorBoardCallback]
        )

    # Run the training
    trainer.train()