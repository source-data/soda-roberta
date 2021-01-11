from pathlib import Path
from typing import List, Tuple, Dict
from xml.etree.ElementTree import parse, Element, ElementTree, fromstring, tostring
from math import floor
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from tokenizers import Encoding
from transformers import RobertaTokenizerFast, BatchEncoding
import regex as re
from .encoder import XMLEncoder
from .xmlcode import (
    CodeMap, SourceDataCodes as sd
)
from common.utils import innertext, progress
from common import TOKENIZER_PATH, NER_DATASET
from common.config import config


class Preparator:
    """Processes source xml documents into examples that can be used in a token classification task.
    It tokenizes the text with the provided tokenizer. 
    The XML is used to generate labels according to the provided CodeMap.
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_dir_path (Path):
            The path to the source xml files.
        dest_dir_path (Path):
            The path of the destination directory where the files with the encoded labeled examples should be saved.
        tokenizer (ByteLevelBPETokenizer):
            The pre-trained tokenizer to use for processing the inner text.
        code_maps (List[CodeMap)]:
            A list of CodeMap, each specifying Tthe XML-to-code mapping of label codes to specific combinations of tag name and attribute values.
        max_length (int):
            Maximum number of token in one example. Examples will be truncated.
        split_ratio (Dict[str, float]):
            Proportion of examples in train, eval and test subsets.
    """
    def __init__(
        self,
        source_dir_path: Path,
        dest_dir_path: Path,
        tokenizer: RobertaTokenizerFast,
        code_maps: List[CodeMap],
        max_length: int = config.max_length
    ):
        self.source_dir_path = source_dir_path
        self.dest_dir_path = dest_dir_path
        self.code_maps = code_maps
        self.max_length = max_length
        self.tokenizer = tokenizer
        assert self._dest_dir_is_empty(), f"{self.dest_dir_path} is not empty! Will not overwrite pre-existing dataset."

    def _dest_dir_is_empty(self) -> bool:
        if self.dest_dir_path.exists():
            # https://stackoverflow.com/a/57968977
            return not any([True for _ in self.dest_dir_path.iterdir()])
        else:
            return True

    def run(self, ext: str = 'xml'):
        """Runs the coding and labeling of xml examples.
        Saves the resulting text files to the destination directory.

        Args:
            ext (str):
               The extension (WITHOUT THE DOT) of the files to be coded.
        """
        labeled_examples = []
        filepaths = list(self.source_dir_path.glob(f"**/*.{ext}"))
        for i, filepath in enumerate(filepaths):
            progress(i, len(filepaths), f"{filepath.name}                 ")
            with filepath.open() as f:
                xml_example: ElementTree = parse(f)
            xml_example: Element = xml_example.getroot()
            tokenized, token_level_labels = self._encode_example(xml_example)
            labeled_examples.append({
                'tokenized': tokenized,
                'label_ids': token_level_labels
            })
        self._save_json(labeled_examples)
        return labeled_examples

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml)
        inner_text = innertext(xml_encoder.element)
        tokenized: BatchEncoding = self.tokenizer(
            inner_text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        token_level_labels = {}
        for code_map in self.code_maps:
            xml_encoded = xml_encoder.encode(code_map)
            labels = self._align_labels(tokenized, xml_encoded, code_map, inner_text)
            token_level_labels[code_map.name] = labels
        return tokenized, token_level_labels

    def _align_labels(self, tokenized: Encoding, xml_encoded: Dict, code_map: CodeMap, inner_text) -> List[int]:
        num_tokens = len(tokenized.input_ids)
        # prefill with outside of entity label 'O' using IOB2 scheme
        token_level_labels = ['O'] * num_tokens
        # tokenizer may have truncated the example
        last_token_start, last_token_end = tokenized.offset_mapping[-2]  # -2 because the last token is </s> with offsets (0,0) by convention
        # get the character-level start end end of the xml element and try to map to tokens
        inner_text = inner_text[:last_token_end]
        for element_start, element_end in xml_encoded['offsets']:
            # check we are still within the truncated example
            if (element_start <= last_token_start) & (element_end < last_token_end):
                code = xml_encoded['label_ids'][element_start]  # element_end would give the same, maybe check with assert
                assert xml_encoded['label_ids'][element_start] == xml_encoded['label_ids'][element_end - 1], f"{xml_encoded['label_ids'][element_start:element_end]}\n{element_start, element_end}"
                start_token_idx = self._char_to_token(element_start, inner_text, tokenized)
                end_token_idx = self._char_to_token(element_end, inner_text, tokenized)
                # sanity check
                try:
                    assert start_token_idx is not None, f"\n\nproblem with start token None."
                    assert end_token_idx is not None, f"\n\nproblem with end token None."
                except Exception:
                    import pdb; pdb.set_trace()
                if (start_token_idx == end_token_idx):
                    # In addition, the tokenizer may generate a token that is actually spanning an element boundary.
                    # Recursive tokenized in the XMLEncode is NOT a solution as it will force learning on atypical tokenization that will not be
                    # representative of tokenization of free text. It actually destroys prediction :-(
                    # But empty element cannot not correspond to any token
                    start, end = tokenized.offset_mapping[end_token_idx]
                    if (start <= element_start) and (end > element_end):
                        print(f"WARNING: token overlaps element boundary {code_map.constraints[code]['tag']} at position {element_end} in '{inner_text[start-10:start]}>>>{inner_text[start:element_end]}^{inner_text[element_end:end]}...<<<{inner_text[end:end+10]}'")
                        # if next token outside of an element, will be labeled; if part of next element, labelig will be overriden
                        end_token_idx += 1 if end_token_idx <= num_tokens else num_tokens
                    else:
                        print(f"WARNING: emtpy element {code_map.constraints[code]['tag']}? at position {element_start, element_end} in >>>{inner_text[element_start:element_start+50]}...<<<")
                prefix = "B"  # for B-egining token according to IOB2 scheme
                if code_map.mode == 'whole_entity':  # label all the tokens corresponding to the xml element
                    for token_idx in range(start_token_idx, end_token_idx):
                        label = self._int_code_to_iob2_label(prefix, code, code_map)
                        token_level_labels[token_idx] = label
                        prefix = "I"  # for subsequet I-nside tokens
                elif code_map.mode == 'boundary_start' and (start_token_idx != end_token_idx):  # label the B-egining of non-empty elements
                    label = self._int_code_to_iob2_label(prefix, code, code_map)
                    token_level_labels[start_token_idx] = label
            else:
                # the last token has been reached, no point scanning further elemnts
                break
        return token_level_labels

    @staticmethod
    def _char_to_token(element_pos, inner_text, tokenized):
        # Nasty: because of RobertaTokenizer's behavior with spaces, 
        # a space before a word is included in token. When this happens across xml element boundary, 
        # the character at the boundary position is a space and is included in the next or previous token outside the element.
        # In addition, BatchEncoding.char_to_token() will return None if the token is a single space
        # proper token will be found only from next or previous character, respectively
        # This gymnastic is to try to circumven this.
        pos = element_pos
        # _, last_pos = tokenized.offset_mapping[-2]  # end of last non special token
        if pos >= len(inner_text):
            token_idx = len(tokenized.input_ids)
            return token_idx
        elif inner_text[pos] != ' ':  # usual case, not in a space, all fine
            token_idx = tokenized.char_to_token(pos)
            return token_idx
        while (inner_text[pos] == ' ') and (pos < len(inner_text) - 1): pos += 1  # scanning for non space on the right
        if inner_text[pos] == ' ':  # we are still in a run of space and at the end of the string!
            token_idx = len(tokenized.input_ids) - 1
        else:
            # __.token    is tokenized into two single spaces plus one .token (dot is spacial character produced by RobertaTokenizer)
            #    ^        need to scan until non space character
            # 5           element_start = 5
            # 5678        pos = 8 after scanning
            # 234         actual start_token_idx 2, first non space token is 4, tokens 2 and 3 are single spaces
            num_single_space_tokens = pos - 1 - element_pos
            try:
                token_idx = tokenized.char_to_token(pos) - num_single_space_tokens
            except Exception:
                import pdb; pdb.set_trace()
        return token_idx

    @staticmethod
    def _int_code_to_iob2_label(prefix: str, code: int, code_map: CodeMap) -> str:
        label = code_map.constraints[code]['label']
        iob2_label = f"{prefix}-{label}"
        return iob2_label

    def _save_json(self, examples: List):
        if not self.dest_dir_path.exists():
            self.dest_dir_path.mkdir()
        # saving line by line to json-line file
        filepath = self.dest_dir_path / "data.jsonl"
        with filepath.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
            for example in examples:
                j = {
                    'tokens': example['tokenized'].tokens(),
                    'input_ids': example['tokenized'].input_ids,
                    'label_ids':  example['label_ids']
                }
                f.write(f"{json.dumps(j)}\n")

    def verify(self):
        filepaths = self.dest_dir_path.glob("**/*.jsonl")
        for p in filepaths:
            with p.open() as f:
                for n, line in enumerate(f):
                    j = json.loads(line)
                    L = len(j['tokens'])
                    assert L <= self.max_length, f"Length verification: error line {n} in {p} with {len(j['tokens'])} tokens > {self.max_length}."
                    assert len(j['input_ids']) == L, f"mismatch in number of tokens and input_ids: error line {n} in {p}"
                    for k, label_ids in j['label_ids'].items():
                        assert len(label_ids) == L, f"mismatch in number of tokens and {k} label_ids: error line {n} in {p}"
        print("\nLength verification: OK!")
        return True


def self_test():
    # example = '<sd-panel> of an adult <sd-tag type="gene">Prox1</sd-tag>-<sd-tag type="gene">Cre</sd-tag><sd-tag type="gene">ER</sd-tag>T2;<sd-tag type="gene">Ilk</sd-tag>+/+ <sd-tag type="organism">mouse</sd-tag> (referred to as "Adult Control")</sd-panel>'
    example = "<xml>Here <sd-panel>it is<sd-tag role='reporter'> </sd-tag>: <i>nested <sd-tag role='reporter'>in</sd-tag> <sd-tag category='entity' type='gene' role='intervention'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End </sd-panel>."
    example += ' 1 2 3 4 5 6 7 8 9 0' + '</xml>'  # to test truncation
    path = Path('/tmp/test_dataprep')
    path.mkdir()
    source_path = path / 'source'
    source_path.mkdir()
    dest_dir_path = path / 'dataset'
    source_file_path = source_path / 'example.xml'
    source_file_path.write_text(example)
    max_length = 20  # in token!
    expected_tokens = [
        '<s>',
        'Here', 'Ġit', 'Ġis', 'Ġ:', 'Ġnested', 'Ġin',
        'ĠCre', 'b', '-', '1',
        'Ġwith', 'Ġsome',
        'Ġtail',
        '.', 'ĠEnd', 'Ġ.',
        'Ġ1', 'Ġ2',
        '</s>'
    ]
    expected_label_codes = {
        'entity_types': [
            'O',
            'O', 'O', 'O', 'O', 'O', 'O',
            'B-GENEPROD', 'I-GENEPROD', 'I-GENEPROD', 'I-GENEPROD',
            'O', 'O',
            'B-GENEPROD',
            'O', 'O', 'O',
            'O',
            'O',
            'O'
        ],
        'geneprod_roles': [
            'O',
            'O', 'O', 'O', 'O', 'O', 'O',
            'B-CONTROLLED_VAR', 'I-CONTROLLED_VAR', 'I-CONTROLLED_VAR', 'I-CONTROLLED_VAR',
            'O', 'O',
            'B-MEASURED_VAR',
            'O', 'O', 'O',
            'O',
            'O',
            'O'
        ],
        'boring': [
            'O',
            'O', 'O', 'O', 'O', 'O', 'B-BORING',
            'O', 'O', 'O', 'O',
            'O', 'O',
            'O',
            'O', 'O', 'O',
            'O',
            'O',
            'O'
        ],
        'panel_start': [
            'O',
            'O', 'B-PANEL_START', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O',
            'O', 'O',
            'O',
            'O', 'O', 'O',
            'O',
            'O',
            'O'
        ]
    }
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        data_prep = Preparator(source_path, dest_dir_path, tokenizer, [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.PANELIZATION], max_length=max_length)
        labeled_examples = data_prep.run()
        print("\nXML examples:")
        print(example)
        print("\nLabel codes: ")
        for i in range(len(labeled_examples[0]['tokenized'].input_ids)):
            token = labeled_examples[0]['tokenized'].tokens()[i]
            input_id = labeled_examples[0]['tokenized'].input_ids[i]
            decoded = tokenizer.decode(input_id)
            label_ids = "\t".join([labels[i] for labels in labeled_examples[0]['label_ids'].values()])
            print(f"{token}\t{decoded}\t{label_ids}")
        labeled_example_label_ids = labeled_examples[0]['label_ids']
        assert labeled_examples[0]['tokenized'].tokens() == expected_tokens, labeled_examples[0]['tokenized'].tokens()
        assert labeled_example_label_ids['entity_types'] == expected_label_codes['entity_types'], labeled_example_label_ids['entity_types']
        assert labeled_example_label_ids['geneprod_roles'] == expected_label_codes['geneprod_roles'], labeled_example_label_ids['geneprod_roles']
        assert labeled_example_label_ids['panel_start'] == expected_label_codes['panel_start'], labeled_example_label_ids['panel_start']
        assert data_prep.verify()
        filepath = dest_dir_path / "data.jsonl"
        print(f"\nContent of saved file ({filepath}):")
        with filepath.open() as f:
            for line in f:
                j = json.loads(line)
                print(json.dumps(j))
    finally:
        shutil.rmtree('/tmp/test_dataprep/')
        print("cleaned up and removed /tmp/test_corpus")
    print("Looks like it is working!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepares the conversion of xml documents into datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", nargs="?", help="Directory where the xml files are located.")
    parser.add_argument("dest_dir", nargs="?", default=NER_DATASET, help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir_path = args.source_dir
    if source_dir_path:
        code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.BORING, sd.PANELIZATION]
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        dest_dir_path = args.dest_dir
        dest_dir_path = Path(dest_dir_path)
        source_dir_path = Path(source_dir_path)
        for subset in ["train", "eval", "test"]:
            print(f"Preparing: {subset}")
            sdprep = Preparator(source_dir_path / subset, dest_dir_path / subset, tokenizer, code_maps)
            sdprep.run()
            sdprep.verify()
        print("\nDone!")
    else:
        self_test()