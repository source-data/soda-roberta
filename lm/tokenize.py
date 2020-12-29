from argparse import ArgumentParser
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from . import DATASET, TOKENIZER_PATH, DEFAULT_TOKENIZER_NAME


def main():
    parser = ArgumentParser(description="Try the tokenizer.")
    parser.add_argument("text", help="Text to tokenize.")
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER_NAME, help="The path to the tokenizer files.")
    args = parser.parse_args()
    tokenizer_name = args.tokenizer_name
    text = args.text

    tokenizer = ByteLevelBPETokenizer(
        f"{TOKENIZER_PATH}/{tokenizer_name}-vocab.json",
        f"{TOKENIZER_PATH}/{tokenizer_name}-merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    tokenized = tokenizer.encode(text)
    print(tokenized.tokens)
    print(tokenized.ids)
    print(tokenized.offsets)


if __name__ == '__main__':
    main()