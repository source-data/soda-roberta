from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, RobertaTokenizerFast
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

class Tagger:

    def __init__(
        self,
        tokenizer,
        panel_model,
        ner_model,
        geneprod_role_model=None,
        small_mol_role_model=None,
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

    def __call__(self, examples: Union[str, List[str]]) -> str:
        if isinstance(examples, list):
            return self._pipeline(examples)
        else:
            return self._pipeline([examples])

    def _pipeline(self, examples: Union[List[str], str]) -> List:

        panelize_output = self._panelize_pipeline(examples)

        ner_output, geneprod_mask, small_mol_mask = self._ner_pipeline(examples, panelize_output)

        pipeline_output = self._roles_pipeline(examples, ner_output, geneprod_mask, small_mol_mask)

        serialized = json.dumps(pipeline_output, ensure_ascii=False, indent=2)

        return serialized
        

    def _panelize_pipeline(self, examples: Union[List[str], str]):

        panel_pipe = LongTextTokenClassificationPipeline(
            task="token-classification",
            model=self.panel_model,
            tokenizer=self.tokenizer,
            device=torch.device(0 if torch.cuda.is_available() else "cpu"),
            aggregation_strategy="simple"
        )

        panelized = panel_pipe("Figure " + examples[0] if isinstance(examples, list) else "Figure " + examples, stride=50)

        panelize_output = self._panelize_postprocess(panelized, examples[0] if isinstance(examples, list) else examples)

        return panelize_output

    def _ner_pipeline(self, examples: Union[List[str], str], panelize_output: List[Dict[str, Any]]):
        ner_pipe = LongTextTokenClassificationPipeline(
            task="token-classification",
            model=self.ner_model,
            tokenizer=self.tokenizer,
            device=torch.device(0 if torch.cuda.is_available() else "cpu"),
            aggregation_strategy="simple"
        )

        ner_tagged = ner_pipe(examples[0] if isinstance(examples, list) else examples, stride=100)

        ner_output, geneprod_mask, small_mol_mask = self._ner_postprocess(
            ner_tagged, 
            panelize_output,
            examples[0] if isinstance(examples, list) else examples
            )

        return ner_output, geneprod_mask, small_mol_mask

    def _roles_pipeline(self, examples: Union[List[str], str], ner_output: List[Dict[str, Any]], geneprod_mask: List[Tuple[int,int]], small_mol_mask: List[Tuple[int,int]]) -> List[Dict[str, Any]]:
        geneprod_output = self._get_roles(
            examples[0] if isinstance(examples, list) else examples,
            geneprod_mask,
            self.geneprod_role_model
        )

        small_mol_output = self._get_roles(
            examples[0] if isinstance(examples, list) else examples,
            small_mol_mask,
            self.small_mol_role_model
        )

        pipeline_output = self._roles_post_process(ner_output, geneprod_output, small_mol_output)

        return pipeline_output

    def _tokenize(self, examples: List[str]) -> Dict[str, List[int]]:
        tokenized = self.tokenizer(
            examples,
            return_attention_mask=False,
            return_special_tokens_mask=True,
            # return_tensors='pt',
            truncation=True
        )
        return tokenized

    def _panelize_postprocess(self, inputs: List[Dict[str, Any]], original_sentence: str) -> List[Dict[str, Any]]:
        # TODO I should generate a dataclass for the output of this function

        panelize_output = []
        did_panel_start = False
        restore_original = len("Figure ")
        group_text = ""

        for n, entity in enumerate(inputs):
            # Dealing with the first panel_group, typically empty
            if n == 0:
                if (entity["entity_group"] == "O"):
                    if entity["end"]-restore_original >= 0:
                        panelize_output.append(
                            {
                                "panel": 0,
                                "text": original_sentence[entity["start"]: entity["end"]-restore_original],
                                "start": max(0, entity["start"]-restore_original),
                                "end": entity["end"]-restore_original,
                            }
                        )
                    
                if (entity["entity_group"] == "PANEL_START"):   
                    did_panel_start = True
                    start = max(0, entity["start"]-restore_original)
            else:
                if entity["entity_group"] == "PANEL_START":
                    did_panel_start = True
                    start = max(0, entity["start"]-restore_original)
                    panel_group = len(panelize_output)

                if (entity["entity_group"] == "O") and did_panel_start:
                    end = entity["end"]-restore_original
                    panelize_output.append(
                        {
                            "panel": panel_group,
                            "text": original_sentence[start: end],
                            "start": start,
                            "end": end,
                        }
                    )

                    did_panel_start = False

        return panelize_output

    def _ner_postprocess(self, entities: List[Dict[str, Any]], panels: List[Dict[str, Any]], sentence: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int,int]], List[Tuple[int,int]]]:
        ner_output = []
        geneprod_masked = []
        small_mol_masked = []
        for panel in panels:
            panel_entities = []
            for entity in entities:
                if entity["word"] not in ["-"]:
                    if (entity['start'] >= panel['start']) and (entity['start'] < panel['end']) and (entity['end'] <= panel['end']):
                        if entity['entity_group'] != 'O':
                            if entity['entity_group'] in ["EXP_ASSAY", "DISEASE"]:
                                categories = {"EXP_ASSAY": "assay", "DISEASE": "disease"}
                                panel_entities.append(
                                    {
                                        "text": sentence[entity["start"]: entity["end"]],
                                        "type": "",
                                        "category": categories.get(entity["entity_group"], ""),
                                        "type_score": "",
                                        "category_score": float(entity["score"]),
                                        "start": entity["start"],
                                        "end": entity["end"]
                                    }
                                )
                            else:
                                if entity["entity_group"] == "GENEPROD":
                                    geneprod_masked.append((entity["start"], entity["end"]))
                                if entity["entity_group"] == "SMALL_MOLECULE":
                                    small_mol_masked.append((entity["start"], entity["end"]))
                                panel_entities.append(
                                    {
                                        "text": sentence[entity["start"]: entity["end"]],
                                        "type": entity["entity_group"],
                                        "category": "",
                                        "type_score": float(entity["score"]),
                                        "category_score": "",
                                        "start": entity["start"],
                                        "end": entity["end"]
                                    }
                                )
            panel["entities"] = panel_entities
            ner_output.append(panel)


        return (ner_output, geneprod_masked, small_mol_masked)

    def _get_roles(self, sentence: str, mask: List[Tuple[int,int]], model: AutoModelForTokenClassification) -> List[Dict[str, Any]]:
        masked_sentence = self._get_masked_sentence(sentence, mask)
        roles_pipe = LongTextTokenClassificationPipeline(
            task="token-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device(0 if torch.cuda.is_available() else "cpu"),
            aggregation_strategy="simple"
        )
        roles_output = roles_pipe(masked_sentence)
        roles_clean = []
        for result in roles_output:
            masks_together = result["word"].count("[MASK]")
            if masks_together == 0:
                continue
            elif masks_together == 1:
                roles_clean.append(result)
            else:
                roles_clean.append(result)
                for _ in range(masks_together-1):
                    roles_clean.append(result)

        assert len(roles_clean) == len(mask)
        for entity, role in zip(mask, roles_clean):
            role["start"] = entity[0]
            role["end"] = entity[1]
        return roles_clean


    @staticmethod
    def _get_masked_sentence(sentence: str, mask: List[Tuple[int,int]]) -> str:
        sentence_list = list(sentence)
        ranges_to_mask = []
        output_string = ""

        if mask == []:
            output_string = sentence
        else:
            for idx, (start, end) in enumerate(mask):
                if len(mask)==1:
                    output_string += sentence[0: start] + "[MASK]" + sentence[end: ]
                else:
                    if idx == 0:
                        output_string += sentence[0: start] + "[MASK]"
                        prev_end = end
                    elif idx == len(mask)-1:
                        output_string += sentence[prev_end: start] + "[MASK]" + sentence[end: ]
                    else:
                        output_string += sentence[prev_end: start] + "[MASK]"
                        prev_end = end

        #         ranges_to_mask.append(idx)
        # for idx, char in enumerate(sentence_list):
        #     if idx not in ranges_to_mask:
        #         output_list.append(char)
        #     elif (idx in ranges_to_mask) and( output_list[-1] != "[MASK]"):
        #         output_list.append("[MASK]")
        #     elif (idx in ranges_to_mask) and( output_list[-1] == "[MASK]"):
        #         continue
        #     else:
        #         raise NotImplementedError

        return output_string

    @staticmethod
    def _roles_post_process(ner, geneprods, small_mols):
        for panel in ner:
            for entity in panel["entities"]:
                if entity["type"] == "GENEPROD":
                    for geneprod in geneprods:
                        if (geneprod["start"] == entity["start"]) and (geneprod["end"] == entity["end"]):
                            entity["role"] = geneprod['entity_group']
                            entity["role_score"] = float(geneprod['score'])

                if entity["type"] == "SMALL_MOLECULE":
                    for small_mol in small_mols:
                        if (small_mol["start"] == entity["start"]) and (small_mol["end"] == entity["end"]):
                            entity["role_score"] = float(small_mol['score'])
                            entity["role"] = small_mol['entity_group']
        return ner


class SmartTagger(Tagger):

    def __init__(
        self,
        tokenizer_source: str = "michiyasunaga/BioLinkBERT-large",
        panelizer_source: str = "EMBO/sd-panelization-v2",
        ner_source: str = "EMBO/sd-ner-v2",
        geneprod_roles_source: str = "EMBO/sd-geneprod-roles-v2",
        small_mol_roles_source: str = "EMBO/sd-smallmol-roles-v2",
        add_prefix_space: bool = True,
    ):

        self.add_prefix_space = add_prefix_space
        super().__init__( 
            AutoTokenizer.from_pretrained(tokenizer_source, 
                                        add_prefix_space=self.add_prefix_space
                                        ),
            AutoModelForTokenClassification.from_pretrained(panelizer_source),
            AutoModelForTokenClassification.from_pretrained(ner_source),
            AutoModelForTokenClassification.from_pretrained(geneprod_roles_source),
            AutoModelForTokenClassification.from_pretrained(small_mol_roles_source),
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
        # TODO! I should generate a dataclass for this! ! !

        return grouped_entities

        
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

