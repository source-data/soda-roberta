from transformers import pipeline
from argparse import ArgumentParser
from common import LM_MODEL_PATH
from common.config import config


def main():
    parser = ArgumentParser(description="Try to use model for mask filling.")
    parser.add_argument("text", nargs="?", default="Let us try this model to see if it <mask>.", help="The text to be sumitted. Use '<mask>' as masking token.")
    parser.add_argument("--model_path", default=LM_MODEL_PATH, help="The path to the mode files.")
    args = parser.parse_args()
    text = args.text
    model_path = args.model_path
    fill_mask = pipeline(
        "fill-mask",
        model=model_path,
        tokenizer=config.tokenizer
    )
    result = fill_mask(text)
    print(f"{result}")


if __name__ == "__main__":
    main()
