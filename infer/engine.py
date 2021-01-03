from transformers import (
    pipeline, TokenClassificationPipeline, RobertaForTokenClassification, RobertaTokenizer
)
from tokenizers import ByteLevelBPETokenizer
from argparse import ArgumentParser
import json
from typing import List, Dict
from common import NER_MODEL_PATH


class Engine:

    def __init__(self, model, tokenizer, serializer):
        self.model: RobertaForTokenClassification = model
        self.tokenizer: ByteLevelBPETokenizer = tokenizer
        self.serializer: Serializer = serializer
        self.ner: TokenClassificationPipeline = pipeline(
            task='ner',
            tokenizer=self.tokenizer,
            model=self.model,
            use_fast=False
        )

    def tag(self, input: str) -> str:
        tagged: List[Dict] = self.ner(input)
        # A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
        #     the corresponding input, or each entity if this pipeline was instantiated with
        #     :obj:`grouped_entities=True`) with the following keys:

        #     - **word** (:obj:`str`) -- The token/word classified.
        #     - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
        #     - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
        #       `grouped_entities` is set to True.
        #     - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
        #       corresponding token in the sentence.
        #     - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
        #       Only exists if the offsets are available within the tokenizer
        #     - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
        #       Only exists if the offsets are available within the tokenizer
        serialized: str = self.serialize(tagged)
        return serialized


class Serializer:

    def to_json(self, tagged):
        j = tagged
        return json.dumps(j, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tags text.")
    parser.add_argument("text", nargs="?", default="This is Creb1 speaking", help="Directory where the xml files are located.")
    args = parser.parse_args()
    model = RobertaForTokenClassification.from_pretrained({NER_MODEL_PATH})
    pre_trained_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    pre_trained_tokenizer.save_pretrained("/tmp/pre_trained_tokenizer")
    tokenizer = ByteLevelBPETokenizer.from_file(
        "/tmp/pre_trained_tokenizer/vocab.json",
        "/tmp/pre_trained_tokenizer/merges.txt"
    )
    serializer = Serializer()
    engine = Engine(model, tokenizer, serializer)