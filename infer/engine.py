from transformers import (
    pipeline, TokenClassificationPipeline, BatchEncoding
    RobertaForTokenClassification, RobertaTokenizerFast, RobertaConfig
)
from argparse import ArgumentParser
import json
import torch
import numpy as np
from typing import List, Dict
from common import NER_MODEL_PATH


class Engine:

    def __init__(self, model, tokenizer, serializer):
        self.tokenizer: RobertaTokenizerFast = tokenizer
        self.model: RobertaForTokenClassification = model
        self.serializer: Serializer = serializer
        # self.ner: TokenClassificationPipeline = pipeline(
        #     task='ner',
        #     tokenizer=self.tokenizer,
        #     model=self.model,
        #     use_fast=False
        # )

    def tag(self, text: str) -> str:
        # tagged: List[Dict] = self.ner(input)
        # what follows is from TokenClassificationPipeline but avoiding transforming labels_idx into str
        tokens = self.tokenizer(
            text,
            return_attention_mask=False,
            return_tensors='pt',
            truncation=True,
        )
        self.predict(tokens)
        serialized: str = self.serializer(tagged, format='str')
        return serialized

    def predict(self, tokens: BatchEncoding) -> List[Dict]:
        with torch.no_grad():
            # tokens = self.ensure_tensor_on_device(**tokens)
            entities = self.model(**tokens)[0][0].cpu().numpy()
            input_ids = tokens["input_ids"].cpu().numpy()[0]
        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)  # why not softmax(-1) in torhc before np?
        labels_idx = score.argmax(axis=-1)
        filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if (self.model.config.id2label[label_idx] not in self.ignore_labels) and not special_tokens_mask[idx]
            ]
        for idx, label_idx in filtered_labels_idx:
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            # check tokenizer.decode()clean_up_tokenization_spaces  https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase
            entity = {
                "word": word,
                "score": score[idx][label_idx].item(),
                "label_idx": label_idx
            }
            entities += [entity]


class Serializer:

    def __call__(self, input: List[Dict], format: str = "json") -> str:
        if format == "json":
            return self.to_json(input)
        else:
            return input

    def to_json(self, tagged):
        def new_entity(iob: str) -> Dict:
            label = iob[2:]
            constraint = self.codes.from_label(label)
            entity = {
                'tag': constraint['tag'],
                'text': t['word'],  # godamn special character!!
                'score': {
                    constraint['tag']: t['score']  # preparing when there will be multiple models
                },
            }
            for attr, val in constraint['attributes'].items():
                entity[attr] = val[0]
            return entity

        j = {
            'smtag': [
                {
                    'entities': []
                }
            ]
        }
        entity = {}
        for t in tagged:
            idx = t['label_idx']
            iob = self.codes.iob2_labels[idx]  # IOB2 label
            if iob != "O":
                prefix = iob[0:1]
                if prefix == "B-":  # begining of a new entity
                    entity = new_entity(iob)
                    # while entity
                elif prefix == "I-" and entity:  # inside an entity, continue with previous entity_label
                    current_entity_label = iob[2:]
                    assert current_entity_label == label  # sanity check
                    entity[text] += entity['word']
                    # no further change since
                # if I- inside entity predicted bu no B- before, 
                # it would be a mistake, with inference missing the begining of an entity. 
                # Not sure if it happens but if it would, then create new entity as if begining.
                elif prefix == "I-" and not entity:
                    entity = new_entity(iob)
                else:
                    # somethign is wrong...
                    print(f"Something is wrong at token {t}.")
                    raise
            else:
                if entity and entity not in j['smtag']['entities']:
                    j['smtag']['entities'].append(new_entity)
                    entity = {}
        return json.dumps(j, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tags text.")
    parser.add_argument("text", nargs="?", default="This is Creb1 speaking", help="Directory where the xml files are located.")
    args = parser.parse_args()
    model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/checkpoint-1200")
    text = args.text
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    serializer = Serializer()
    engine = Engine(tokenizer, model, serializer)
    tagged = engine.tag(text)
    print(tagged)
