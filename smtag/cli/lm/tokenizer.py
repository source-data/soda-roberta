from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from os.path import exists
from tqdm import tqdm
import json
import numpy as np

def get_training_corpus(list_):
    dataset = list_
    for start_idx in range(0, len(dataset), 50):
        samples = dataset[start_idx : start_idx + 50]
        yield samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates tokenizer based on 13GB of full PubMed articles and 12 million abstract and figure captions from OAPMC.")
    parser.add_argument("source_dir", help="Directory where the source files are located.")
    parser.add_argument("base_tokenizer", nargs="?", help="Based tokenizer from which a biomedical version will be created.")
    parser.add_argument("--full_articles_path", default="", help="jsonl file with the full body text of articles.")
    parser.add_argument("--vocab_size", default=0, help="total size of the vocabularry.")
    args = parser.parse_args()
    source_dir_path = args.source_dir
    base_tokenizer = args.base_tokenizer
    full_articles_path = args.full_articles_path

    old_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    vocab_size = old_tokenizer.vocab_size if args.vocab_size == 0 else args.vocab_size
    print(vocab_size)

    oapmc_abstracts_figs = []

    for split in tqdm(["train", "test", "eval"]):
        assert exists(f'{source_dir_path}/{split}.txt') == True, f"The file is not found in the path: {base_tokenizer}/{split}.txt"
        with open(f'{source_dir_path}/{split}.txt') as f:
            lines = f.readlines()
            for line in lines:
                oapmc_abstracts_figs.append(line)

    if full_articles_path != "":
        with open(full_articles_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            if np.random.random() < 0.1:
                oapmc_abstracts_figs.append(json.loads(json_str)["body"])


    training_corpus = get_training_corpus(oapmc_abstracts_figs)

    new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size)

    new_tokenizer.save_pretrained("excel-roberta-ls")