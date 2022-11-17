from datasets import load_dataset
import json
from math import ceil
from tqdm import tqdm
import numpy as np

import logging

logger = logging.getLogger('smtag.excell_roberta.model')
logging.basicConfig( level='INFO' )
if __name__ == "__main__":


    data_folder = '/app/data/text/oapmc_abstracts_figs/'
    abstracts_figs = [data_folder+"eval.txt", data_folder+"test.txt", data_folder+"train.txt"]

    max_words_sentence = 256

    eval_list = []

    logging.info("Reading the full article list from pubmed")

    with open('/app/data/text/pubmed-full/pubmed-articles.jsonl', 'r') as json_file:
        json_list = list(json_file)
        
    with open("/app/data/text/excell-roberta/train.txt", "a") as train_file:                    
        count_pubmed_full_articles = 0
        count_words = 0
        
        # Pubmed full text data
        
        for article in tqdm(json_list):
            count_pubmed_full_articles += 1
            article_length = len(json.loads(article)["body"].split())
            count_words += article_length
            total_splits = ceil(article_length / max_words_sentence)
            if np.random.random() < 0.95:
                train_file.write(json.dumps({"text": json.loads(article)["body"]}) + "\n")
            else:
                eval_list.append(json.loads(article)["body"])

    del json_file
        
    print(50 * "*")        
    print(f"A file with {count_pubmed_full_articles} full text PMC papers has been created:")
    print(f"It contains {count_words} words.")
    print(50 * "*")    

    logging.info("Loading the datasets from HuggingFace")

    bc5che = load_dataset("EMBO/BLURB", "BC5CDR-chem-IOB")
    bc5dis = load_dataset("EMBO/BLURB", "BC5CDR-disease-IOB")
    bc2gm = load_dataset("EMBO/BLURB", "BC2GM-IOB")
    ncbi = load_dataset("EMBO/BLURB", "NCBI-disease-IOB")
    jnlpba = load_dataset("EMBO/BLURB", "JNLPBA")

    with open("/app/data/text/excell-roberta/train.txt", "a") as train_file:                    
        for dataset in [bc5che, bc5dis, bc2gm, ncbi, jnlpba]:
            for split in ['train', 'validation', 'test']:
                for line in tqdm(dataset[split]['tokens']):
                    if "DOCSTART" not in ' '.join(line):
                        count_words += len(line)
                        if np.random.random() < 0.95:
                            train_file.write(json.dumps({"text": ' '.join(line)}) + "\n")
                        else:
                            eval_list.append(' '.join(line))

    del bc5che
    del bc5dis
    del bc2gm
    del ncbi
    del jnlpba

    print(50 * "*")        
    print(f"A file with {count_pubmed_full_articles} full text PMC papers has been created:")
    print(f"It contains {count_words} words.")
    print(50 * "*")        
    logging.info("Reading the abstracts and figures from OAPMC")

    for filename in abstracts_figs:
        with open(filename, 'r') as abstract_file:
            abstract_figs_list = abstract_file.readlines()
        
        with open("/app/data/text/excell-roberta/train.txt", "a") as train_file:                    
            # Abstracts and figures from the source data dataset
            count_abstract_figs = 0
            for split in tqdm(abstract_figs_list):
                for line in split:
                    count_abstract_figs += 1
                    count_words += len(line.split())
                    if np.random.random() < 0.99:
                        train_file.write(json.dumps({"text": line}) + "\n")
                    else:
                        eval_list.append(line)
            
        del abstract_file
        del abstract_figs_list
            
    with open("/app/data/text/excell-roberta/eval.txt", "a") as eval_file: 
        for line in eval_list:
            eval_file.write(json.dumps({"text": line}) + "\n")

    print(50 * "*")        
    print(f"A file with {count_pubmed_full_articles} full text PMC papers has been created:")
    print(f"It contains {count_words} words.")
    print(f"It contains {count_abstract_figs} abstracts and figures from open access PMC.")
    print(50 * "*")        