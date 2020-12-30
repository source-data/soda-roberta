from pathlib import Path
from argparse import ArgumentParser
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from .config import config
from . import (
    DATASET, TOKENIZER_PATH
)


def main():
    parser = ArgumentParser(description="Training tokenizer on text files.")
    parser.add_argument("text_dir", nargs="?", default=DATASET, help="Path to the directory containgin the text files.")
    parser.add_argument("-t", "--tokenizer_path", default=TOKENIZER_PATH, help="Path to the saved trained tokenizer.")
    args = parser.parse_args()
    text_dir = args.text_dir
    tokenizer_path = args.tokenizer_path
    if Path(tokenizer_path).exists():
        paths = [str(x) for x in Path(text_dir).glob("**/*.txt")]
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.pre_tokenizer = Whitespace
        tokenizer.train(
            files=paths,
            vocab_size=config.vocab_size,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ]
        )
        tokenizer.save_model(tokenizer_path)
    else:
        print(f"{tokenizer_path} does not exists, will not be able to save tokenizer. Create dir first and re-run the command.")


if __name__ == '__main__':
    main()
