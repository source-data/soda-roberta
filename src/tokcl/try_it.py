from argparse import ArgumentParser
from transformers import pipeline, RobertaTokenizerFast, RobertaForTokenClassification
from common import TOKCL_MODEL_PATH

_EXAMPLE = """<s> F. Western blot of input and eluates of Upf1 domains purification in a Nmd4-HA strain. The band with the # might corresponds to a dimer of Upf1-CH,
 bands marked with a star correspond to residual signal with the anti-HA antibodies (Nmd4). Fragments in the eluate have a smaller size because the protein A pa
rt of the tag was removed by digestion with the TEV protease. G6PDH served as a loading control in the input samples </s>"""

if __name__ == "__main__":
    parser = ArgumentParser(description="Quick try of a NER model")
    parser.add_argument("text", nargs="?", default=_EXAMPLE, help="Text to analyze.")
    parser.add_argument("-M", "--model-path", default={TOKCL_MODEL_PATH}, help="Path to the model.")

    args = parser.parse_args()
    text = args.text
    model_path = args.model_path
    model = RobertaForTokenClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    pipe = pipeline('ner', model, tokenizer=tokenizer)
    res = pipe(text)
    for r in res:
        print(r['word'], r['entity'])
