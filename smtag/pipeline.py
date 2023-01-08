from transformers import (
    AutoModelForTokenClassification, AutoTokenizer
)
import numpy as np
import json
import torch
from typing import Optional, List, Tuple, Union, Any, Dict
from .xml2labels import CodeMap, SourceDataCodes as sd
from transformers.pipelines.token_classification import (TokenClassificationArgumentHandler, 
                                                         TokenClassificationPipeline, AggregationStrategy )

from transformers import Pipeline
from torch import Tensor
from transformers.pipelines.base import ChunkPipeline
import tensorflow as tf
import numpy as np
from transformers.models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from transformers.models.bert.tokenization_bert import BasicTokenizer
from math import ceil
import warnings

class Entity:

    def __init__(
        self,
        input_id: int,
        ner_label: str,
        role_geneprod_label: str,
        role_small_mol_label: str,
        ner_code_map: CodeMap,
        geneprod_role_code_map: CodeMap,
        small_mol_role_code_map: CodeMap,
        type_score: List[float],
        geneprod_role_score: List[float],
        molecule_role_score: List[float],
        
    ):
        self.input_ids = [input_id]
        self.ner_label = ner_label
        self.role_geneprod_label = role_geneprod_label
        self.role_small_mol_label = role_small_mol_label
        self.type_score = type_score
        self.geneprod_role_score = geneprod_role_score
        self.molecule_role_score = molecule_role_score
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

    def to_dict(self, tokenizer):
        self.text = tokenizer.decode(self.input_ids)
        d = {'text': self.text.strip()}  # removes the leading space from the RobertaTokenizer
        for k, v in self.attrib.items():
            d[k] = v
        try:
            d["type"] = d.pop("entity_type")
        except KeyError:
            pass

        d["type_score"] = np.array(self.type_score).mean()

        if d.get("type", "") in ["geneprod", "gene", "protein"]:
            d["role_score"] = np.array(self.geneprod_role_score).mean()

        if d.get("type", "") in ['molecule']:
            d["role_score"] = np.array(self.molecule_role_score).mean()
        return d


class Serializer:

    def __init__(
        self,
        tokenizer,
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
            j_panel_group = {'panel_group': {
                "entities": []
            }
            }
            ner_results, geneprod_roles_results, small_mol_roles_results, panel_text = panel_group
            num_panels = len(ner_results['input_ids'])
            for panel_idx in range(num_panels):
                label_ids = ner_results['input_ids'][panel_idx]
                entity_types = ner_results['labels'][panel_idx]
                special_tokens_mask = ner_results['special_tokens_mask'][panel_idx]
                entity_geneprod_roles = geneprod_roles_results['labels'][panel_idx]
                entity_small_mol_roles = small_mol_roles_results['labels'][panel_idx]
                ner_scores = ner_results['scores'][panel_idx]
                geneprod_roles_scores = geneprod_roles_results['scores'][panel_idx]
                small_mol_roles_scores = small_mol_roles_results['scores'][panel_idx]
                entity_list = []
                entity = None
                for i in range(len(label_ids)):
                    try:
                        special_token = special_tokens_mask[i]
                    except IndexError:
                        special_token = 1
                    # ignore positions that are special tokens
                    if special_token == 0:
                        input_id_ner = label_ids[i]
                        label_ner = entity_types[i]
                        score_ner = ner_scores[i]
                        score_geneprod_roles = geneprod_roles_scores[i]
                        score_small_mol_roles = small_mol_roles_scores[i]
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
                                    entity_dict = entity.to_dict(self.tokenizer)
                                    # This is the place where scores should be added to the entity

                                    entity_list.append(entity_dict)
                                entity = Entity(
                                    input_id=input_id_ner,
                                    ner_label=label_ner,
                                    role_geneprod_label=iob_geneprod_role[2:] if iob_geneprod_role != "O" else "",
                                    role_small_mol_label=iob_small_mol_role[2:] if iob_small_mol_role != "O" else "",
                                    ner_code_map=self.ner_code_map,
                                    geneprod_role_code_map=self.geneprod_role_code_map,
                                    small_mol_role_code_map=self.small_mol_role_code_map,
                                    type_score=[score_ner],
                                    geneprod_role_score=[score_geneprod_roles],
                                    molecule_role_score=[score_small_mol_roles],
                                )
                            elif prefix == "I-" and entity is not None:  # already inside an entity, continue add token ids
                                entity.input_ids.append(input_id_ner)
                                entity.type_score.append(score_ner)
                                entity.geneprod_role_score.append(score_geneprod_roles)
                                entity.molecule_role_score.append(score_small_mol_roles)
                            # if currently I-nside and entity predicted but no prior B-eginging detected.
                            # Would be an inference mistake.
                            elif prefix == "I-" and entity is None:  # should not happen, but who knows...
                                # It is happening sometimes, giving text like ##token
                                # Simply passing avoids it 
                                # Avoding this would leave a cleaner graph.
                                pass
                            else:
                                # something is wrong...
                                print(f"Something is wrong at token {input_id_ner} with IOB label {iob_ner}.")
                                print(j)
                                raise Exception("serialization failed")
                        elif entity is not None:
                            entity_list.append(entity.to_dict(self.tokenizer))
                            entity = None
                j_panel_group['panel_group']['entities'].append(entity_list)
                j_panel_group['panel_group']['panel_text'] = panel_text
            j['smtag'].append(j_panel_group)
        return json.dumps(j, indent=2)


class Tagger:

    def __init__(
        self,
        tokenizer,
        panel_model,
        ner_model,
        geneprod_role_model,
        small_mol_role_model,
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
        # tokenized = self._tokenize(examples)
        # panelized = self.panelize(tokenized)

        pipe = LongTextTokenClassificationPipeline(task="token-classification",
                            model=self.panel_model,
                            tokenizer=self.tokenizer,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            aggregation_strategy="simple")

        panelized = [pipe("Figure " + examples[0] if isinstance(examples, list) else "Figure " + examples, stride=50)]
        
        for panel_group in panelized[0]:
            panel_group = self._truncate(panel_group)
            ner_results = self.ner(panel_group)
            geneprod_roles_results = self.roles(ner_results, ['B-GENEPROD', 'I-GENEPROD'], self.geneprod_role_model)
            small_mol_roles_results = self.roles(ner_results, ['B-SMALL_MOLECULE', 'I-SMALL_MOLECULE'], self.small_mol_role_model)
            panel_text = self.tokenizer.decode(panel_group["input_ids"][0])
            tagged.append((ner_results, geneprod_roles_results, small_mol_roles_results, panel_text))
        print(tagged)

        serialized = self.serializer(tagged, format='json')

        self.serialized = serialized
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
        for input_ids, labels in zip(predictions['input_ids'], predictions['labels']):
            panel_group = {
                'input_ids': [],
                # 'attention_mask': [],
                'special_tokens_mask': []
            }
            panel = []
            for input_id, label in zip(input_ids, labels):
                # need to skip beginging and end of sentence tokens
                if input_id not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
                    # are we at the start of a new panel?
                    if label == panel_start_label_code:
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
        return self.predict(inputs, self.ner_model)

    def roles(
        self,
        inputs: Dict[str, List[int]],
        labels_to_mask: List[str],
        role_model
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

    def predict(self, inputs: Dict[str, List[int]], model) -> Dict[str, List[int]]:
        # what follows is inspired from TokenClassificationPipeline but avoiding transforming labels_idx into str
        model.eval()
        # pad lists to same length and tensorify
        if isinstance(inputs["input_ids"], (list, tuple)):
            examples = self.tokenizer.pad(
                inputs,
                return_tensors="pt",
                padding=True,  # this will pad to the max length in the batch
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
    
    def _truncate(self, panel_group):
        new_inputs = {}
        for key, value in panel_group.items():
            if key == "input_ids":
                new_inputs[key] = [value[:self.panel_model.config.max_position_embeddings-1] + [self.tokenizer.sep_token_id]]
            else:
                new_inputs[key] = [value[:self.panel_model.config.max_position_embeddings]]
        return new_inputs

class SmartTagger(Tagger):

    def __init__(
        self,
        tokenizer_source: str = "michiyasunaga/BioLinkBERT-large",
        panelizer_source: str = "EMBO/sd-panelization-v2",
        ner_source: str = "EMBO/sd-ner-v2",
        geneprod_roles_source: str = "EMBO/sd-geneprod-roles-v2",
        small_mol_roles_source: str = "EMBO/sd-smallmol-roles-v2",
        add_prefix_space: bool = True
    ):
        self.add_prefix_space = add_prefix_space
        super().__init__( 
            AutoTokenizer.from_pretrained(tokenizer_source, 
                                          is_pretokenized=True, 
                                          add_prefix_space=self.add_prefix_space
                                          ),
            AutoModelForTokenClassification.from_pretrained(panelizer_source),
            AutoModelForTokenClassification.from_pretrained(ner_source),
            AutoModelForTokenClassification.from_pretrained(geneprod_roles_source),
            AutoModelForTokenClassification.from_pretrained(small_mol_roles_source)
        )

class LongTextTokenClassificationPipeline(ChunkPipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.
    Strings of any length can be passed. If they exceed `ModelForTokenClassification.config.max_position_embeddings` tokens,
    they will be divided into several parts text that will be passed to the `forward` method.
    The results will then be concatenated together and be sent back.
    *LongTextTokenClassificationPipeline* uses `offsets_mapping` and therefore is available only with `FastTokenizer`.
    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    """    
    default_input_names = "sequences"

    def __init__(self, args_parser=TokenClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        if not self.tokenizer.is_fast:
            raise TypeError(
            """LongTextTokenClassificationPipeline works only with fast tokenizers.
            Please choose a fast tokenizer."""
            )

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        stride: Optional[int] = None,
    ):

        preprocess_params = {}
        if offset_mapping is not None:
            preprocess_params["offset_mapping"] = offset_mapping

        postprocess_params = {}
        if grouped_entities is not None or ignore_subwords is not None:
            if grouped_entities and ignore_subwords:
                aggregation_strategy = AggregationStrategy.FIRST
            elif grouped_entities and not ignore_subwords:
                aggregation_strategy = AggregationStrategy.SIMPLE
            else:
                aggregation_strategy = AggregationStrategy.NONE

            if grouped_entities is not None:
                warnings.warn(
                    "`grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )
            if ignore_subwords is not None:
                warnings.warn(
                    "`ignore_subwords` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )

        if aggregation_strategy is not None:
            if isinstance(aggregation_strategy, str):
                aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]
            if (
                aggregation_strategy
                in {AggregationStrategy.FIRST, AggregationStrategy.MAX, AggregationStrategy.AVERAGE}
                and not self.tokenizer.is_fast
            ):
                raise ValueError(
                    "Slow tokenizers cannot handle subwords. Please set the `aggregation_strategy` option"
                    'to `"simple"` or use a fast tokenizer.'
                )
            postprocess_params["aggregation_strategy"] = aggregation_strategy
        if ignore_labels is not None:
            postprocess_params["ignore_labels"] = ignore_labels
            
            
        if stride is not None:
            if not isinstance(stride, int): 
                raise TypeError(
                    f"Strides must be of type `int`. {type(stride)} was given."
                )
            postprocess_params["stride"] = stride
            preprocess_params["stride"] = stride
            
        return preprocess_params, {}, postprocess_params
    
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.
        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of texts) for token classification.
        Return:
            A list or a list of list of `dict`: Each result comes as a list of dictionaries (one for each token in the
            corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
            the following keys:
            - **word** (`str`) -- The token/word classified. This is obtained by decoding the selected tokens. If you
              want to have the exact string in the original sentence, use `start` and `stop`.
            - **score** (`float`) -- The corresponding probability for `entity`.
            - **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
              *aggregation_strategy* is not `"none"`.
            - **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
              token in the sentence.
            - **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
            - **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mapping = self._args_parser(inputs, **kwargs)
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping
            
        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence, offset_mapping=None, stride=0):
        truncation = False
        
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=None,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )

#         sentence_chunks = self._get_sentence_chunks(model_inputs["input_ids"], stride)
                        
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping
                    
        model_inputs["sentence"] = sentence
        
        idx_lookup = list(range(len(model_inputs["input_ids"])))[1:-1]
        first_token = 0
        bos_token = model_inputs["input_ids"][0]
        eos_token = model_inputs["input_ids"][-1]
        
        chunk_inputs = {}
        
        while first_token < len(idx_lookup):
            start = max(0,first_token-stride)
            end = min(start + self.model.config.max_length - 2, len(idx_lookup))
            
            chunk_inputs["input_ids"] = self._to_tensor(
                [bos_token] + model_inputs["input_ids"][1:-1][start:end] + [eos_token]
                )
            chunk_inputs["token_type_ids"] = self._to_tensor(
                [0] + model_inputs["token_type_ids"][1:-1][start:end] + [0]
                )
            chunk_inputs["attention_mask"] = self._to_tensor(
                [1] + model_inputs["attention_mask"][1:-1][start:end] + [1]
                )
            chunk_inputs["special_tokens_mask"] = self._to_tensor(
                [1] + model_inputs["special_tokens_mask"][1:-1][start:end] + [1]
                )
            chunk_inputs["offset_mapping"] = [(0,0)] + model_inputs["offset_mapping"][1:-1][start:end] + [(0,0)]
            chunk_inputs["chunk_sentence"] = self.tokenizer.decode(chunk_inputs["input_ids"][0])
            chunk_inputs["sentence"] = sentence
            
            first_token = end
                        
            yield {**chunk_inputs}
            
    def _forward(self, chunk_inputs: Dict[str, Any]) -> List[dict]:
        # Forward
        special_tokens_mask = chunk_inputs.pop("special_tokens_mask")
        offset_mapping = chunk_inputs.pop("offset_mapping", None)
        sentence = chunk_inputs.pop("sentence")
        chunk_sentence = chunk_inputs.pop("chunk_sentence")
        if self.framework == "tf":
            logits = self.model(chunk_inputs.data)[0]
        else:
            logits = self.model(**chunk_inputs)[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "chunk_sentence": chunk_sentence,
            **chunk_inputs,
        }
    
    def postprocess(self, model_outputs: List[Dict[str, Any]], 
                    aggregation_strategy=AggregationStrategy.NONE, 
                    ignore_labels=None, 
                    stride=0):
        sentence = model_outputs[0]["sentence"]
        aggregated_tokenizer_outputs = self.tokenizer(sentence,
            return_tensors=self.framework,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )
        input_ids = aggregated_tokenizer_outputs["input_ids"]
        offset_mapping = aggregated_tokenizer_outputs["offset_mapping"]
        special_tokens_mask = aggregated_tokenizer_outputs["special_tokens_mask"]
        
        logits = self._aggregate_chunk_outputs(model_outputs, stride)
        
        if ignore_labels is None:
            ignore_labels = ["O"]
        logits = logits.numpy()
        input_ids = input_ids[0]
        offset_mapping = offset_mapping[0] if offset_mapping is not None else None
        special_tokens_mask = special_tokens_mask[0].numpy()

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
        
        pre_entities = self.gather_pre_entities(
            sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
        )
        grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
        
        # Here we get the right output from the model. 
        # The problem comes from the lines just

        # Add first group even in it is "O"

        # If panel start then add panel_start + O

        # Append already tokenized
        
        panel_group_text = []
        did_panel_start = False
        for n, entity in enumerate(grouped_entities):
            if n == 0:
                if (entity["entity_group"] == "O"):
                    panel_group_text.append(
                        self.tokenizer(
                            entity["word"],
                            return_special_tokens_mask=True,
                            return_token_type_ids=False,
                            return_attention_mask=False
                        )
                    )
                if (entity["entity_group"] == "PANEL_START"):   
                    did_panel_start = True
                    group_text = entity["word"] 
            else:
                if entity["entity_group"] == "PANEL_START":
                    did_panel_start = True
                    group_text = entity["word"]
                if (entity["entity_group"] == "O") and did_panel_start:
                    group_text = group_text + " " + entity["word"]
                    panel_group_text.append(
                            self.tokenizer(
                            group_text,
                            return_special_tokens_mask=True,
                            return_token_type_ids=False,
                            return_attention_mask=False
                        )
                    )
                    did_panel_start = False
                    group_text=""

        # did_panel_start = False
        # if len(grouped_entities) == 1:
        #     panel_group_text.append(grouped_entities[0]["word"])
        # else:
        #     for entity in grouped_entities:
        #         if entity["entity_group"] == "PANEL_START":
        #             did_panel_start = True
        #         if (entity["entity_group"] == "O") and did_panel_start:
        #             panel_group_text.append(entity["word"])
        #             did_panel_start = False

        # if panel_group_text == []:
        #     panel_group_text = [sentence]

        # panel_groups = self.tokenizer(panel_group_text,
        #                         return_special_tokens_mask=True,
        #                         return_token_type_ids=False,
        #                         return_attention_mask=False)

        return panel_group_text
        
    def _to_tensor(self, inputs: List[Any]) -> Union[tf.Tensor, torch.tensor, np.ndarray]:
        if self.framework == "pt":
            return torch.tensor(inputs).unsqueeze(0)
        if self.framework == "tf":
            return tf.reshape(tf.convert_to_tensor(inputs), (1,-1))
        if self.framework == "np":
            return np.array(inputs).reshape(1,-1)

    def _aggregate_chunk_outputs(self, outputs: 
                                 List[Dict[str, Any]], 
                                 stride: int) -> Union[tf.Tensor, torch.tensor, np.ndarray]:
        """
        Change this to numpy or lits to save cuda space
        """
        for iter_, chunk_output in enumerate(outputs):
            is_first = (iter_ == 0)
            is_last = (iter_ == len(outputs)-1)
            if is_first:
                logits = chunk_output["logits"][0][:-1]
            elif is_last:
                logits = self._concat(logits, chunk_output["logits"][0][stride+1:])
            else:
                logits = self._concat(logits, chunk_output["logits"][0][stride+1:-1])
                
        return logits
            
    def _concat(self, 
                 t1: Union[tf.Tensor, torch.tensor, np.ndarray],
                 t2: Union[tf.Tensor, torch.tensor, np.ndarray],
                 axis: int  = 0
                ) -> Union[tf.Tensor, torch.tensor, np.ndarray]:
        if self.framework == "pt":
            concat = torch.concat([t1, t2], axis=axis)
        if self.framework == "tf":
            concat = tf.concat([t1, t2], axis=axis)
        if self.framework == "np":
            concat = np.concatenate([t1, t2], axis=axis)
        return concat
    
    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()
                    else:
                        start_ind = int(start_ind.numpy())
                        end_ind = int(end_ind.numpy())
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    if aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        warnings.warn("Tokenizer does not support real words, using fallback heuristic", UserWarning)
                    is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.
        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.
        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.
        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups
