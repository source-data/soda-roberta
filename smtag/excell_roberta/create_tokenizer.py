from transformers import AutoTokenizer
import argparse
from typing import List
from tokenizers import (AddedToken, normalizers, pre_tokenizers, Tokenizer)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import RobertaProcessing
from tokenizers.decoders import ByteLevel
import glob
from transformers import RobertaTokenizerFast
import os

class ExcellRobertaTokenizer:
    """
    Tokenizer class similar to RoBERTa, but with biomedical vocabulary.
    """
    def __init__(self, tokenizer_name, vocab_size: int = 52000, min_frequency: int = 50):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer_name = tokenizer_name

    def create_model(self, files: List[str]):
        tok = Tokenizer(BPE())

        tok.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(), 
            ]
        )

        tok.add_special_tokens(
            [
                '<s>', 
                '</s>', 
                '<unk>', 
                '<pad>', 
                AddedToken("<mask>", rstrip=False, lstrip=False, single_word=False, normalized=False),
            ]
        )

        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_preffix_space=True)
            # [
            #     pre_tokenizers.ByteLevel(), 
            #     pre_tokenizers.WhitespaceSplit(), 
            #     pre_tokenizers.Digits(individual_digits=self.individual_digits), 
            #     pre_tokenizers.Punctuation()
            # ]

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
            show_progress=True,
            continuing_subword_prefix="##"
        )
        tok.train(files=files, trainer=trainer)

        tok.post_processor = RobertaProcessing(
            cls=("<s>", tok.token_to_id("<s>")),
            sep=("</s>", tok.token_to_id("</s>")),
        )

        tok.decoder = ByteLevel()

        encoding = tok.encode("Let's test this tokenizer.")
        print("This is an example of the tokenizer in the sentence: Let's test this tokenizer.")
        print(encoding.tokens)

        tok.save("tokenizer_model.json")

        excell_roberta = RobertaTokenizerFast(tokenizer_file="tokenizer_model.json", model_max_length=512)        
        excell_roberta.save_pretrained(self.tokenizer_name)
        os.remove("tokenizer_model.json") 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates tokenizer based on 13GB of full PubMed articles and 12 million abstract and figure captions from OAPMC.")
    parser.add_argument("tokenizer_name", nargs="?", help="name of the new tokenizer.")
    parser.add_argument("--file_name", action='append', help="Path to the file to be used to build the tokenizer.")
    parser.add_argument("--vocab_size", default=52000, type=int, help="total size of the vocabularry.")
    parser.add_argument("--min_freq", default=50, type=int, help="Minimum times a token must appear to be considered.")
    args = parser.parse_args()
    file_name = args.file_name
    assert isinstance(args.file_name, list)
    tokenizer_name = args.tokenizer_name
    vocab_size = args.vocab_size
    min_freq = args.min_freq

    new_tokenizer = ExcellRobertaTokenizer(
        tokenizer_name, 
        vocab_size=vocab_size, 
        min_frequency=min_freq, 
    )
    
    new_tokenizer.create_model(file_name)