from transformers import (
    TokenClassificationPipeline, BatchEncoding,
    RobertaForTokenClassification, RobertaTokenizerFast
)
from argparse import ArgumentParser
import json
import torch
import numpy as np
from typing import List, Dict
from tokcl.xmlcode import CodeMap, SourceDataCodes as sd
from common import NER_MODEL_PATH


class Entity:

    def __init__(self, input_id: int, ner_label: str, role_label: str, ner_code_map: CodeMap, role_code_map: CodeMap):
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

    def to_dict(self, tokenizer: RobertaTokenizerFast):
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

    def __call__(self, *args, format: str = "json") -> str:
        if format == "json":
            return self.to_json(*args)
        else:
            return input

    def to_json(self, pred_types: List[Dict[str, List]], pred_roles: List[Dict[str, List]]) -> str:

        j = {'smtag': []}
        for entity_types, entity_roles in zip(pred_types, pred_roles):
            entity_list = []
            entity = None
            for i in range(len(entity_types['input_ids'])):
                input_id_ner = entity_types['input_ids'][i]
                idx_ner = entity_types['labels_idx'][i]
                iob_ner = self.ner_code_map.iob2_labels[idx_ner]  # convert label idx into IOB2 label
                idx_role = entity_roles['labels_idx'][i]
                iob_role = self.role_code_map.iob2_labels[idx_role]
                if iob_ner != "O":  # begining or inside an entity, for ex B-GENEPROD
                    prefix = iob_ner[0:2]  # for example B-
                    label_ner = iob_ner[2:]  # for example GENEPROD
                    if prefix == "B-":  # begining of a new entity
                        if entity is not None:  # save current entity before creating new one
                            entity_list.append(entity.to_dict(self.tokenizer))
                        entity = Entity(
                            input_id=input_id_ner,
                            ner_label=label_ner,
                            role_label=iob_role[2:] if iob_role != "O" else "",
                            ner_code_map=self.ner_code_map,
                            role_code_map=self.role_code_map
                        )
                    elif prefix == "I-" and entity is not None:  # already inside an entity, continue add token ids
                        entity.input_ids.append(input_id_ner)
                    # if currently I-nside and entity predicted but no prior B-eginging detected.
                    # Would be an inference mistake.
                    elif prefix == "I-" and entity is None:  # should not happen, but who knows...
                        entity = Entity(
                            input_id=input_id_ner,
                            ner_label=label_ner,
                            role_label=iob_role[2:] if iob_role != "O" else "",
                            ner_code_map=self.ner_code_map,
                            role_code_map=self.role_code_map
                        )
                    else:
                        # something is wrong...
                        print(f"Something is wrong at token {input_id_ner} with IOB label {iob_ner}.")
                        print(j)
                        raise Exception("serialization failed")
                elif entity is not None:
                    entity_list.append(entity.to_dict(self.tokenizer))
                    entity = None
            j['smtag'].append(entity_list)
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
        self.panel_code_map = sd.PANELIZATION
        self.ner_code_map = sd.ENTITY_TYPES
        self.role_code_map = sd.GENEPROD_ROLES
        self.serializer = Serializer(
            self.ner_code_map,
            self.role_code_map,
            self.tokenizer,  # for decoding
        )

    def pipeline(self, text: str) -> str:
        output = self._tokenize(text)
        panelized: List[BatchEncoding] = self.panelize(output)
        entity_types: List[Dict[List]] = self.ner(panelized)
        geneprod_roles: List[Dict[List]] = self.roles(entity_types)
        serialized = self.serializer(entity_types, geneprod_roles, format='json')
        return serialized

    def _tokenize(self, text) -> BatchEncoding:
        tokens = self.tokenizer(
            text,
            return_attention_mask=False,
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return tokens

    def panelize(self, input: BatchEncoding) -> List[BatchEncoding]:
        labeled = self.predict([input], self.panel_model)
        labeled = labeled[0]
        panel_start_label_code = self.panel_code_map.iob2_labels.index('B-PANEL_START')
        panels = []
        panel = []
        for input_id in labeled['input_ids']:
            if input_id == panel_start_label_code:
                if panel:
                    encoding = self._tokenize(self.tokenizer.decode(panel))  # nicely return tensors and special tokens
                    panels.append(encoding)
                    panel = []
            panel.append(input_id)
        # don't forget to include last accumulated panel
        if panel:
            encoding = self._tokenize(self.tokenizer.decode(panel))
            panels.append(encoding)
        return panels

    def ner(self, input: List[BatchEncoding]) -> List[Dict[str, List]]:
        output = self.predict(input, self.ner_model)
        return output

    def roles(self, input) -> List[Dict[str, List]]:
        mask_token_id = self.tokenizer.mask_token_id
        geneprod_codes = [
            self.ner_code_map.iob2_labels.index('B-GENEPROD'),
            self.ner_code_map.iob2_labels.index('I-GENEPROD'),
        ]
        masked_input = []
        for panel in input:
            masked_input_ids = []
            for i in range(len(panel['input_ids'])):
                input_id = panel['input_ids'][i]
                label_idx = panel['labels_idx'][i]
                if label_idx in geneprod_codes:
                    input_id = mask_token_id
                masked_input_ids.append(input_id)
            # tensorify
            masked_input_ids = torch.tensor(masked_input_ids).unsqueeze(0)
            special_tokens_mask = torch.tensor(panel['special_tokens_mask']).unsqueeze(0)
            masked_panel = {
                'input_ids': masked_input_ids,
                'special_tokens_mask': special_tokens_mask,
            }
            masked_input.append(masked_panel)
        # tensorify
        output = self.predict(masked_input, self.role_model)
        return output

    def predict(self, input: List[BatchEncoding], model: RobertaForTokenClassification) -> List[Dict[str, List]]:
        # what follows is taken from TokenClassificationPipeline but avoiding transforming labels_idx into str
        # returning a List[Dict[List]] instead of List[List[Dict]] to faciliate serial input-output predictions
        # TODO handle this as a batch rather than one by one
        output = []
        for tokens in input:
            with torch.no_grad():
                # pop() to remove from tokens which are submittted to module.forward()
                special_tokens_mask = tokens.pop("special_tokens_mask").cpu()[0]
                # tokens = self.ensure_tensor_on_device(**tokens)
                entities = model(**tokens)[0][0].cpu()
                input_ids = tokens["input_ids"][0].cpu()
                scores = entities.softmax(-1)
                labels_idx = entities.argmax(-1)
            special_tokens_mask = special_tokens_mask.numpy()
            labels_idx = labels_idx.numpy()
            entities = entities.numpy()
            scores = scores.numpy()
            labels_idx = [(idx, label_idx) for idx, label_idx in enumerate(labels_idx)]
            entities = {
                "input_ids": [],
                "scores": [],
                "labels_idx": []
            }
            for idx, label_idx in labels_idx:
                input_id = int(input_ids[idx])
                score = scores[idx][label_idx].item()
                entities["input_ids"].append(input_id)
                entities["scores"].append(score)
                entities["labels_idx"].append(label_idx)
            entities["special_tokens_mask"] = special_tokens_mask  # restore special_tokens_mask for potential carry over to next serial model
            output.append(entities)
        return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Tags text.")
    parser.add_argument("text", nargs="?", default="We studies mice with genetic ablation of the ERK1 gene in brain and muscle.", help="Directory where the xml files are located.")
    args = parser.parse_args()
    text = args.text
    panel_model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/PANELIZATION")
    ner_model = RobertaForTokenClassification.from_pretrained(f"{NER_MODEL_PATH}/NER/")
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
