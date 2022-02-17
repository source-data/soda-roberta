import pdb
from transformers import (
    RobertaForTokenClassification, RobertaTokenizerFast
)

import json
import torch
from typing import List, Dict, Tuple, Union
from .xml2labels import CodeMap, SourceDataCodes as sd


class Entity:

    def __init__(
        self,
        input_id: int,
        ner_label: str,
        role_geneprod_label: str,
        role_small_mol_label: str,
        ner_code_map: CodeMap,
        geneprod_role_code_map: CodeMap,
        small_mol_role_code_map: CodeMap
    ):
        self.input_ids = [input_id]
        self.ner_label = ner_label
        self.role_geneprod_label = role_geneprod_label
        self.role_small_mol_label = role_small_mol_label
        ner_constraint = ner_code_map.from_label(ner_label)
        # keep only first value of attributes for serialization
        self.attrib = {attr: val[0] for attr, val in ner_constraint['attributes'].items()}
        if role_geneprod_label:
            role_constraint = geneprod_role_code_map.from_label(role_geneprod_label)
            self.attrib = {attr: val[0] for attr, val in role_constraint['attributes'].items()}
        if role_small_mol_label:
            role_constraint = small_mol_role_code_map.from_label(role_small_mol_label)
            self.attrib = {attr: val[0] for attr, val in role_constraint['attributes'].items()}
        self.text = ''

    def to_dict(self, tokenizer: RobertaTokenizerFast):
        self.text = tokenizer.decode(self.input_ids)
        d = {'text': self.text.strip()}  # removes the leading space from the RobertaTokenizer
        for k, v in self.attrib.items():
            d[k] = v
        return d


class Serializer:

    def __init__(
        self,
        tokenizer: RobertaTokenizerFast,
        ner_code_map,
        geneprod_role_code_map,
        small_mol_role_code_map
    ):
        self.tokenizer = tokenizer
        self.ner_code_map = ner_code_map
        self.geneprod_role_code_map = geneprod_role_code_map
        self.small_mol_role_code_map = small_mol_role_code_map

    def __call__(self, *args, format: str = "json") -> str:
        if format == "json":
            return self.to_json(*args)
        else:
            return input

    def to_json(self, tagged_panel_groups: List[Tuple[Dict[str, torch.Tensor]]]) -> str:

        j = {'smtag': []}
        for panel_group in tagged_panel_groups:
            j_panel_group = {'panel_group': []}
            ner_results, geneprod_roles_results, small_mol_roles_results = panel_group
            num_panels = len(ner_results['input_ids'])
            for panel_idx in range(num_panels):
                label_ids = ner_results['input_ids'][panel_idx]
                entity_types = ner_results['labels'][panel_idx]
                special_tokens_mask = ner_results['special_tokens_mask'][panel_idx]
                entity_geneprod_roles = geneprod_roles_results['labels'][panel_idx]
                entity_small_mol_roles = small_mol_roles_results['labels'][panel_idx]
                entity_list = []
                entity = None
                for i in range(len(label_ids)):
                    special_token = special_tokens_mask[i]
                    # ignore positions that are special tokens
                    if special_token == 0:
                        input_id_ner = label_ids[i]
                        label_ner = entity_types[i]
                        iob_ner = self.ner_code_map.iob2_labels[label_ner]  # convert label idx into IOB2 label
                        label_geneprod_role = entity_geneprod_roles[i]
                        label_small_mol_role = entity_small_mol_roles[i]
                        iob_geneprod_role = self.geneprod_role_code_map.iob2_labels[label_geneprod_role]
                        iob_small_mol_role = self.small_mol_role_code_map.iob2_labels[label_small_mol_role]
                        if iob_ner != "O":  # begining or inside an entity, for ex B-GENEPROD
                            prefix = iob_ner[0:2]  # for example B-
                            label_ner = iob_ner[2:]  # for example GENEPROD
                            if prefix == "B-":  # begining of a new entity
                                if entity is not None:  # save current entity before creating new one
                                    entity_list.append(entity.to_dict(self.tokenizer))
                                entity = Entity(
                                    input_id=input_id_ner,
                                    ner_label=label_ner,
                                    role_geneprod_label=iob_geneprod_role[2:] if iob_geneprod_role != "O" else "",
                                    role_small_mol_label=iob_small_mol_role[2:] if iob_small_mol_role != "O" else "",
                                    ner_code_map=self.ner_code_map,
                                    geneprod_role_code_map=self.geneprod_role_code_map,
                                    small_mol_role_code_map=self.small_mol_role_code_map
                                )
                            elif prefix == "I-" and entity is not None:  # already inside an entity, continue add token ids
                                entity.input_ids.append(input_id_ner)
                            # if currently I-nside and entity predicted but no prior B-eginging detected.
                            # Would be an inference mistake.
                            elif prefix == "I-" and entity is None:  # should not happen, but who knows...
                                entity = Entity(
                                    input_id=input_id_ner,
                                    ner_label=label_ner,
                                    role_geneprod_label=iob_geneprod_role[2:] if iob_geneprod_role != "O" else "",
                                    role_small_mol_label=iob_small_mol_role[2:] if iob_small_mol_role != "O" else "",
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
                j_panel_group['panel_group'].append(entity_list)
            j['smtag'].append(j_panel_group)
        return json.dumps(j, indent=2)


class Tagger:

    def __init__(
        self,
        tokenizer: RobertaTokenizerFast,
        panel_model: RobertaForTokenClassification,
        ner_model: RobertaForTokenClassification,
        geneprod_role_model: RobertaForTokenClassification,
        small_mol_role_model: RobertaForTokenClassification,
    ):
        self.tokenizer = tokenizer  # for encoding
        self.panel_model = panel_model
        self.ner_model = ner_model
        self.geneprod_role_model = geneprod_role_model
        self.small_mol_role_model = small_mol_role_model
        self.panel_code_map = sd.PANELIZATION
        self.ner_code_map = sd.ENTITY_TYPES
        self.geneprod_role_code_map = sd.GENEPROD_ROLES
        self.small_mol_role_code_map = sd.SMALL_MOL_ROLES
        self.serializer = Serializer(
            self.tokenizer,
            self.ner_code_map,
            self.geneprod_role_code_map,
            self.small_mol_role_code_map
        )

    def __call__(self, examples: Union[str, List[str]]) -> str:
        if isinstance(examples, list):
            return self._pipeline(examples)
        else:
            return self._pipeline([examples])

    def _pipeline(self, examples: List[str]) -> List:
        tagged = []
        tokenized = self._tokenize(examples)
        panelized = self.panelize(tokenized)
        for panel_group in panelized:
            ner_results = self.ner(panel_group)
            geneprod_roles_results = self.roles(ner_results, ['B-GENEPROD', 'I-GENEPROD'], self.geneprod_role_model)
            small_mol_roles_results = self.roles(ner_results, ['B-SMALL_MOLECULE', 'I-SMALL_MOLECULE'], self.small_mol_role_model)
            tagged.append((ner_results, geneprod_roles_results, small_mol_roles_results))
        serialized = self.serializer(tagged, format='json')
        return serialized

    def _tokenize(self, examples: List[str]) -> Dict[str, List[int]]:
        tokenized = self.tokenizer(
            examples,
            return_attention_mask=False,
            return_special_tokens_mask=True,
            # return_tensors='pt',
            truncation=True
        )
        return tokenized

    def panelize(self, inputs: Dict[str, List[int]]) -> List[Dict[str, List[int]]]:
        panel_start_label_code = self.panel_code_map.iob2_labels.index('B-PANEL_START')
        predictions = self.predict(inputs, self.panel_model)
        batch = []
        # each element of the predictions['input_ids'] list is a group of panels
        # each group of panel need to be segmented into individual panels using the predicted B-PANEL_START tag
        for input_ids in predictions['input_ids']:
            panel_group = {
                'input_ids': [],
                # 'attention_mask': [],
                'special_tokens_mask': []
            }
            panel = []
            for input_id in input_ids:
                # need to skip beginging and end of sentence tokens
                if input_id not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
                    # are we at the start of a new panel?
                    if input_id == panel_start_label_code:
                        # have we accumulated input_ids to make a panel?
                        if panel:
                            # it is time to add the panel to the current panel group
                            encoded_panel = self._tokenize(self.tokenizer.decode(panel))  # nicely return tensors and special tokens
                            panel_group['input_ids'].append(encoded_panel['input_ids'])
                            # panel_group['attention_mask'].append(encoded_panel['attention_mask'])
                            panel_group['special_tokens_mask'].append(encoded_panel['special_tokens_mask'])
                            panel = []
                    panel.append(input_id)
            # don't forget to include last accumulated panel
            if panel:
                encoded_panel = self._tokenize(self.tokenizer.decode(panel))
                panel_group['input_ids'].append(encoded_panel['input_ids'])
                # panel_group['attention_mask'].append(encoded_panel['attention_mask'])
                panel_group['special_tokens_mask'].append(encoded_panel['special_tokens_mask'])
                panel = []
            batch.append(panel_group)
        return batch

    def ner(self, inputs: Dict[str, List[int]]) -> Dict[str, List[int]]:
        outputs = self.predict(inputs, self.ner_model)
        return outputs

    def roles(
        self,
        inputs: Dict[str, List[int]],
        labels_to_mask: List[str],
        role_model: RobertaForTokenClassification
    ) -> Dict[str, List[int]]:
        masked_inputs = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "special_tokens_mask": torch.tensor(inputs["special_tokens_mask"], dtype=torch.uint8)
        }
        # tensorify labels
        labels = torch.tensor(inputs["labels"])
        for label in labels_to_mask:
            masking_id = self.ner_code_map.iob2_labels.index(label)
            mask = labels == masking_id
            masked_inputs['input_ids'][mask] = self.tokenizer.mask_token_id
        outputs = self.predict(masked_inputs, role_model)
        return outputs

    def predict(self, inputs: Dict[str, List[int]], model: RobertaForTokenClassification) -> Dict[str, List[int]]:
        # what follows is inspired from TokenClassificationPipeline but avoiding transforming labels_idx into str
        model.eval()
        # pad lists to same length and tensorify
        if isinstance(inputs["input_ids"], (list, tuple)):
            examples = self.tokenizer.pad(
                inputs,
                return_tensors="pt",
                padding=True  # this will pad to the max length in the batch
            )
        else:
            # already as tensor; need to clone before popping things out
            examples = {
                "input_ids": inputs["input_ids"].clone(),
                "special_tokens_mask": inputs["special_tokens_mask"].clone()
            }
        # pop() to remove from examples but keep for later
        special_tokens_mask = examples.pop("special_tokens_mask")
        with torch.no_grad():
            # tokens = self.ensure_tensor_on_device(**tokens)
            try:
                outputs = model(**examples)
            except RuntimeError as e:
                print(e)
                import pdb; pdb.set_trace()
            logits = outputs[0].cpu()  # B x L H
            proba = logits.softmax(-1)  # B x L x H
            input_ids = examples["input_ids"].cpu()
            labels = logits.argmax(-1)  # B x L
            scores = proba.take_along_dim(labels.unsqueeze(-1), -1)  # pytorch v 1.10 !!
            scores.squeeze_(-1)
        predictions = {
            "input_ids": input_ids.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "special_tokens_mask": special_tokens_mask.tolist(),
        }
        return predictions


class SmartTagger(Tagger):

    def __init__(
        self,
        tokenizer_source: str = "roberta-base",
        panelizer_source: str = "EMBO/sd-panels",
        ner_source: str = "EMBO/sd-ner",
        geneprod_roles_source: str = "EMBO/sd-geneprod-roles",
        small_mol_roles_source: str = "EMBO/sd-small-mol-roles"

    ):
        super().__init__(
            RobertaTokenizerFast.from_pretrained(tokenizer_source),
            RobertaForTokenClassification.from_pretrained(panelizer_source),
            RobertaForTokenClassification.from_pretrained(ner_source),
            RobertaForTokenClassification.from_pretrained(geneprod_roles_source),
            RobertaForTokenClassification.from_pretrained(small_mol_roles_source)
        )
