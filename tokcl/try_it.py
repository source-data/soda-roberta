from argparse import ArgumentParser
from transformers import pipeline, RobertaForTokenClassification, RobertaTokenizerFast
from common import NER_MODEL_PATH

if __name__ == "__main__":
    parser = ArgumentParser(description="Quick try of a NER model")
    parser.add_argument("text", nargs="?", default="We studies mice with genetic ablation of the ERK1 gene in brain and muscle.", help="Text to analyze.")
    parser.add_argument("-M", "--model-path", default={NER_MODEL_PATH}, help="Path to the model.")

    args = parser.parse_args()
    text = args.text
    model_path = args.model_path
    model = RobertaForTokenClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    pipe = pipeline(
        'ner',
        model,
        tokenizer=tokenizer
    )
    res = pipe(text)
    for r in res:
        print(r['word'], r['entity'])