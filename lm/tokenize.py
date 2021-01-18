from argparse import ArgumentParser
from transformers import RobertaTokenizerFast
from common import TOKENIZER_PATH
from common.config import config


def main():
    parser = ArgumentParser(description="Try custom trained tokenizer.")
    parser.add_argument("text", nargs="?", default="This is an example.", help="Text to tokenize.")
    args = parser.parse_args()
    text = args.text

    if config.from_pretrained:
        tokenizer = RobertaTokenizerFast.from_pretrained(config.from_pretrained, max_length=config.max_length)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_length=config.max_length)
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    print("Tokenized:")
    print(tokenized)
    print(tokenized.tokens())


if __name__ == '__main__':
    main()
