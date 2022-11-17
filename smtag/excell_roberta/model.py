import argparse

from datasets import load_dataset

import glob
import logging
import os

from smtag.data_classes import TrainingExcellRoberta, TrainingArgumentsTOKCL

from smtag.tb_callback import MyTensorBoardCallback
from smtag.show import ShowExampleLM
from smtag.train.train_tokcl import TrainTokenClassification

from transformers import (DataCollatorForLanguageModeling, HfArgumentParser,
                            RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM,
                            Trainer)

logger = logging.getLogger('smtag.excell_roberta.model')
    
if __name__ == "__main__":

    # The first part of the code would be the argument parsing
    parser = HfArgumentParser(TrainingExcellRoberta, description="Traing script.")
    # parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="Path (absolute) or name to the tokenizer to be used to train the model")
    parser.add_argument("datapath", help="Path (absolute) tothe data that will be used to train the model")
    parser.add_argument("--max_length", default=512, type=int, help="Max length of the model")
    parser.add_argument("--block_size", default=256, type=int, help="Size of the text blocks (in tokens) to generate examples")
    parser.add_argument("--from_checkpoint", default="", type=str, help="Checkpoint from which to initialize the trainingn of the model.")
    parser.add_argument("--init_weights", default=0.02, type=float, help="Range of the initial weights of the model")
    parser.add_argument( '-log',
                        '--loglevel',
                        default='warning',
                        help='Provide logging level. Example --loglevel debug, default=warning' ) 
    training_args, args = parser.parse_args_into_dataclasses()
    logging.basicConfig( level=args.loglevel.upper() )

    # The second part is defining the tokenizer, configuration, and model

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer, max_len=args.max_length)

    config = RobertaConfig(
        #Common model attributes
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        position_embedding_type="absolute",
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        initializer_range=args.init_weights,
        type_vocab_size=1,
        max_position_embeddings=514,
        intermediate_size=3072,
        #Tokenizer parameters
        tokenizer_class=tokenizer.__class__.__name__,
        prefix=None,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        #Class attributes
        model_type="roberta",
    )

    if args.from_checkpoint == "":
        model = RobertaForMaskedLM(config=config)
    else:
        model = RobertaForMaskedLM.from_pretrained(args.from_checkpoint)

    logger.info(100*"*")
    logger.info(f"You are creating a new model {model.__class__.__name__} from scratch")
    logger.info(f"The model has {int(model.num_parameters()/1e6)} million parameters")
    logger.info(print(model))
    logger.info(100*"*")

    # The third part of the code is generating the data
    logger.info("Reading the dataset")

    ds = load_dataset("json", data_files={'train': [os.path.join(args.datapath, "train.jsonl")],
                                                    #os.path.join(args.datapath, "test.jsonl")],
                                        'eval': os.path.join(args.datapath, "eval.jsonl")})
    logger.info(print(ds))

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    logger.info("Tokenizing datasets")
    tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])

    logger.info(ds["train"][0])
    logger.info(tokenized_datasets["train"][0])
    logger.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Concatenating datasets with fix blocksize")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=32,
        num_proc=16,
    )

    logger.info(ds["train"][0])
    logger.info(lm_datasets["train"][0])
    logger.info(tokenizer.decode(lm_datasets["train"][0]["input_ids"]).replace("##",''))
    logger.info(tokenizer.convert_ids_to_tokens(lm_datasets["train"][0]["input_ids"]))
   
    # The fourth part is generating a data collator for masked language modelling
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
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["eval"],
        callbacks=[ShowExampleLM(tokenizer),
                    MyTensorBoardCallback]
        )

    # Run the training
    trainer.train()

    # Generate a fine tuning on SourceData NER to compare with other models and know if we need more training
    logging.info("Fine-tuning excell-roberta on NER task for evaluation")
    # I should load the model
    last_checkpoint = max(glob.glob(os.path.join("/app/excell-roberta-lm/", 'checkpoint*/')), key=os.path.getmtime)

    # Run the Token classification thing with the parameters of the paper
    finetune_trainer = TrainTokenClassification(
        training_args=TrainingArgumentsTOKCL,
        loader_path="EMBO/sd-nlp-non-tokenized",
        task="NER",
        from_pretrained=last_checkpoint,
        model_type="Autoencoder",
        masked_data_collator=False,
        tokenizer=tokenizer,
        add_prefix_space=True,
        ner_labels="all"
    )

    finetune_trainer()

    # Print out the result for NER