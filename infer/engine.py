from transformers import (
    pipeline, TokenClassificationPipeline, BatchEncoding,
    RobertaForTokenClassification, RobertaTokenizerFast
)
from argparse import ArgumentParser
import json
import torch
import numpy as np
from typing import List, Dict
from tokcl.xmlcode import SourceDataCodes, CodeMap
from common import NER_MODEL_PATH


class Serializer:

    def __init__(self, code_map: CodeMap, tokenizer: RobertaTokenizerFast):
        self.code_map = code_map
        self.tokenizer = tokenizer

    def __call__(self, input: List[Dict], format: str = "json") -> str:
        if format == "json":
            return self.to_json(input)
        else:
            return input

    def to_json(self, tagged):
        def new_entity(t, label: str) -> Dict:
            constraint = self.code_map.from_label(label)
            entity = {
                'tag': constraint['tag'],
                'input_ids': [t['input_id']],
                'score': {
                    constraint['tag']: t['score']  # preparing when there will be multiple models
                },
            }
            for attr, val in constraint['attributes'].items():
                entity[attr] = val[0]
            return entity

        def add_entity(entity: Dict):
            # converts toke id into string cleaning up special space charaters
            entity['text'] = self.tokenizer.decode(entity['input_ids'])
            entity.pop('input_ids')
            if entity and entity not in j['smtag'][0]['entities']:
                j['smtag'][0]['entities'].append(entity)

        j = {
            'smtag': [
                {
                    'entities': []
                }
            ]
        }
        entity = {}
        prev_prefix = ''
        prev_label = ''
        for t in tagged:
            idx = t['label_idx']
            iob = self.code_map.iob2_labels[idx]  # IOB2 label
            if iob != "O":  # begining or inside an entity
                prefix = iob[0:2]
                label = iob[2:]
                if prefix == "B-":  # begining of a new entity
                    # before creating new entity, save current one
                    if prev_prefix in ["B-", "I-"]:
                        add_entity(entity)
                    entity = new_entity(t, label)
                elif prefix == "I-" and entity:  # already inside an entity, continue add text to it
                    assert prev_label == label, f"Something is wrong: I- label follows a different label\n{current_entity_label} != {label}"  # sanity check
                    entity['input_ids'].append(t['input_id'])
                # if currently I-nside and entity predicted but no B-eginging detected before,
                # it would be a mistake, with inference missing the begining of an entity. 
                # Not sure if it happens but if it would, then create new entity as if begining.
                elif prefix == "I-" and not entity:
                    entity = new_entity(t, label)
                else:
                    # somethign is wrong...
                    print(f"Something is wrong at token {t} with IOB label {iob}.")
                    print(j)
                    raise Exception("sad")
                prev_prefix = prefix
                prev_label = label
            elif prev_prefix in ["B-", "I-"]:
                add_entity(entity)
                entity = {}
        return json.dumps(j, indent=4)


class Engine:

    def __init__(
        self,
        tokenizer: RobertaTokenizerFast,
        model: RobertaForTokenClassification,
        serializer: Serializer
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.serializer = serializer

    def tag(self, text: str) -> str:
        tokens = self.tokenize(text)
        tagged, input_ids = self.predict(tokens)
        print(self.tokenizer.decode(input_ids))
        serialized: str = self.serializer(tagged, format='json')
        return serialized

    def tokenize(self, text) -> BatchEncoding:
        tokens = self.tokenizer(
            text,
            return_attention_mask=False,
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return tokens

    def predict(self, tokens: BatchEncoding) -> List[Dict]:
        # what follows is from TokenClassificationPipeline but avoiding transforming labels_idx into str
        special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]  # pop() to remove from tokens which are submittted to module.forward()
        with torch.no_grad():
            # tokens = self.ensure_tensor_on_device(**tokens)
            entities = self.model(**tokens)[0][0].cpu().numpy()
            input_ids = tokens["input_ids"].cpu().numpy()[0]
        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)  # why not softmax(-1) in torhc before np?
        labels_idx = score.argmax(axis=-1)
        filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if not special_tokens_mask[idx]
            ]
        entities = []
        for idx, label_idx in filtered_labels_idx:
            input_id = int(input_ids[idx])
            entity = {
                "input_id": input_id,
                "score": score[idx][label_idx].item(),
                "label_idx": label_idx
            }
            entities += [entity]
        return entities, list(input_ids)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tags text.")
    parser.add_argument("text", nargs="?", default="This is Creb1 speaking", help="Directory where the xml files are located.")
    args = parser.parse_args()
    model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/checkpoint-1200")
    text = args.text
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    code_map = SourceDataCodes.ENTITY_TYPES
    serializer = Serializer(code_map, tokenizer)
    engine = Engine(tokenizer, model, serializer)
    tagged = engine.tag(text)
    print(tagged)
