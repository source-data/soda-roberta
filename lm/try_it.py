from transformers import pipeline
from argparse import ArgumentParser
from pathlib import Path
from ..common import MODEL_PATH


def main():
    parser = ArgumentParser(description="Try to use model for mask filling.")
    parser.add_argument("text", nargs="?", default="Let us try this model to see if it <mask>.", help="The text to be sumitted. Use '<mask>' as masking token.")
    args = parser.parse_args()
    text = args.text
    #model_paths = Path(MODEL_PATH).glob("lm-*")
    #model_paths = [p.name for p in model_paths]
    #most_recent = sorted(model_paths, reverse=True)[0]

    fill_mask = pipeline(
        "fill-mask",
        model=f"{MODEL_PATH}",
        tokenizer=f"{MODEL_PATH}"
    )
    result = fill_mask(text)
    print(f"{result}")


if __name__ == "__main__":
    main()
