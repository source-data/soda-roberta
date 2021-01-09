from transformers import (
    TokenClassificationPipeline, BatchEncoding,
    RobertaForTokenClassification, RobertaTokenizerFast
)
from argparse import ArgumentParser
import json
import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
import dataclasses
from tokcl.xmlcode import SourceDataCodes as sd, CodeMap
from common import NER_MODEL_PATH


class Entity:

    def __init__(self, input_id, ner_label, role_label, ner_code_map, role_code_map):
        self.input_ids = [input_id]
        self.ner_label = ner_label
        self.role_label = role_label
        ner_constraint = ner_code_map.from_label(ner_label)
        # keep only first value of attributes for serialization
        self.attrib = {attr: val[0] for attr, val in ner_constraint['attributes'].items()}
        if role_label:
            role_constraint = role_code_map.from_label(role_label)
            self.attrib = {attr: val[0] for attr, val in role_constraint['attributes'].items()}
        self.text = ''

    def to_dict(self, tokenizer):
        self.text = tokenizer.decode(self.input_ids)
        d = {'text': self.text}
        for k, v in self.attrib.items():
            d[k] = v
        return d


class Serializer:

    def __init__(
        self,
        ner_code_map,
        role_code_map,
        tokenizer: RobertaTokenizerFast
    ):
        self.ner_code_map = ner_code_map
        self.role_code_map = role_code_map
        self.tokenizer = tokenizer

    def __call__(self, input: List[List[Dict]], format: str = "json") -> str:
        if format == "json":
            return self.to_json(input)
        else:
            return input

    def to_json(self, input: Dict[str, List[int]]):

        j = {'smtag': []}
        for p in input:
            entity_list = []
            ner_tokens = input['entity_types']
            role_tokens = input['geneprod_roles']
            entity = None
            for i, in range(len(ner_tokens)):
                t_ner = ner_tokens[i]
                idx_ner = t_ner['label_idx']
                iob_ner = self.ner_code_map.iob2_labels[idx_ner]  # convert label idx into IOB2 label
                t_role = role_tokens[i]
                idx_role = t_role['label_idx']
                iob_role = self.role_code_map.iob2_labels[idx_role]
                if iob_ner != "O":  # begining or inside an entity, for ex B-GENEPROD
                    prefix = iob_ner[0:2]  # for example B-
                    ner_label = iob_ner[2:]  # for example GENEPROD
                    if prefix == "B-":  # begining of a new entity
                        if entity is not None:  # save current entity before creating new one
                            entity_list.append(entity)
                        entity = Entity(
                            input_id=t_ner['input_id'],
                            ner_label=ner_label,
                            role_label=iob_role[2:] if iob_role != "O" else "",
                            ner_code_map=self.ner_code_map,
                        )
                    elif prefix == "I-" and entity is not None:  # already inside an entity, continue add token ids
                        entity.input_ids.append(t['input_id'])
                    # if currently I-nside and entity predicted but no prior B-eginging detected.
                    # Would be an inference mistake.
                    elif prefix == "I-" and entity is None:  # should not happen, but who knows...
                        entity = Entity(
                            input_id=t_ner['input_id'],
                            ner_label=ner_label,
                            role_label=iob_role[2:] if iob_role != "O" else "",
                            ner_code_map=self.ner_code_map,
                        )
                    else:
                        # something is wrong...
                        print(f"Something is wrong at token {t} with IOB label {iob_ner}.")
                        print(j)
                        raise Exception("serialization failed")
                elif entity is not None:
                    entity_list.append(entity)
                    entity = None
            j['smtag'].append(entity.to_dict(self.tokenizer))
        return json.dumps(j, indent=4)


class Tagger:

    def __init__(
        self,
        tokenizer: RobertaTokenizerFast,
        panel_model: RobertaForTokenClassification,
        ner_model: RobertaForTokenClassification,
        role_model: RobertaForTokenClassification
    ):
        self.tokenizer = tokenizer  # for encoding
        self.panel_model = panel_model
        self.ner_model = ner_model
        self.role_model = role_model
        self.panel_code_map = sd.PANELIZATION,
        self.ner_code_map = sd.ENTITY_TYPES,
        self.role_code_map = sd.GENEPROD_ROLES
        self.serializer = Serializer(
            self.tokenizer,  # for decoding
            self.panel_code_map,
            self.ner_code_map,
            self.role_code_map
        )

    def pipeline(self, text: str) -> str:
        output = self.tokenize(text)
        panelized: List[BatchEncoding] = self.panelize(output)
        entity_types: List[List[Dict]] = self.ner(panelized, filter_special_tokens=False)
        geneprod_roles: List[Dict] = self.roles(entities)
        serialized = self.serializer({
            'entity_types': entity_types,
            'geneprod_roles': geneprod_roles
        }, format='json')
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

    def panelize(self, input: BatchEncoding) -> List[BatchEncoding]:
        labeled = self.predict([input], self.panel_model)[0]
        panel_start_label_code = self.panel_code_map.from_label('B-PANEL_START')
        panels = []
        token_ids = []
        for e in labeled:
            if e['label_idx'] == panel_start_label_code:
                if token_ids:
                    encoding = self.tokenizer(self.tokenizer.decode(token_ids))  # encode(decode()) produces nice BatchEncoding with special tokens
                    panels.append(encoding)
                    token_ids = []
            token_ids.append(e['input_id'])
        return panels

    def ner(self, input: List[BatchEncoding]) -> List[List[Dict]]:
        output = self.predict(input, self.ner_model)
        return output

    def roles(self, input: List[List[Dict]]) -> List[List[Dict]]:
        # mask geneprod that are not boring
        mask_token_id = self.tokenizer.mask_token_id
        geneprod_codes = self.ner_code_map.from_label('GENEPROD')
        masked = []
        for p in input:
            input_ids = []
            for t in p:
                input_id = t['input_id']
                label_idx = t['label_idx']
                if label_idx == geneprod_codes:
                    input_id = mask_token_id
                input_ids.append(input_id)
            masked.append({'input_ids': input_ids})
        output = self.predict(masked, self.role_model)
        return output

    def predict(self, input: List[BatchEncoding], model: RobertaForTokenClassification, filter_special_tokens: bool = True) -> List[Dict]:
        # what follows is taken from TokenClassificationPipeline but avoiding transforming labels_idx into str
        output = []
        for tokens in input:
            special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]  # pop() to remove from tokens which are submittted to module.forward()
            with torch.no_grad():
                # tokens = self.ensure_tensor_on_device(**tokens)
                entities = model(**tokens)[0][0].cpu().numpy()
                input_ids = tokens["input_ids"][0].cpu().numpy()
                # score = entities.softmax(-1).numpy()
            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)  # why not softmax(-1) in torch before np?
            labels_idx = score.argmax(axis=-1)
            filtered_labels_idx = [
                    (idx, label_idx)
                    for idx, label_idx in enumerate(labels_idx)
                    if (not special_tokens_mask[idx]) and filter_special_tokens
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
            output.append(entities)
        return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Tags text.")
    parser.add_argument("text", nargs="?", default="This is Creb1 speaking", help="Directory where the xml files are located.")
    args = parser.parse_args()
    text = args.text
    panel_model = RobertaTokenizerFast.from_pretrained(f"{NER_MODEL_PATH}/PANELIZATION")
    ner_model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/NER")
    role_model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/ROLES")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tagger = Tagger(
        tokenizer,
        panel_model,
        ner_model,
        role_model
    )
    tagged = tagger.pipeline(text)
    print(tagged)
