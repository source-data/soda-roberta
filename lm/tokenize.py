from argparse import ArgumentParser
# from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from common import TOKENIZER_PATH
from common.config import config


def main():
    parser = ArgumentParser(description="Try the tokenizer.")
    parser.add_argument("text", help="Text to tokenize.")
    args = parser.parse_args()
    text = args.text

    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)
    # tokenizer = ByteLevelBPETokenizer(
    #     f"{TOKENIZER_PATH}/vocab.json",
    #     f"{TOKENIZER_PATH}/merges.txt",
    # )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    tokenized = tokenizer.encode(text)
    print(tokenized.tokens)
    print()
    print(tokenized.ids)
    print()
    print(tokenized.offsets)
    print()
    print(f"length: {len(tokenized.tokens)} token.")


if __name__ == '__main__':
    main()
