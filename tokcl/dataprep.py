from pathlib import Path
from typing import List, Tuple, Dict
from xml.etree.ElementTree import parse, Element, ElementTree
from math import floor
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from tokenizers import Encoding
from transformers import RobertaTokenizerFast, BatchEncoding
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
    """
    def __init__(
        self,
        source_dir_path: Path,
        dest_dir_path: Path,
        tokenizer: RobertaTokenizerFast,
        code_maps: List[CodeMap],
        max_length: int = config.max_length,
        split_ratio: Dict = config.split_ratio
    ):
        self.source_dir_path = source_dir_path
        self.dest_dir_path = dest_dir_path
        self.code_maps = code_maps
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
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
            # self._save_individual_example(filepath, labeled_token)
        split_examples = self._split(labeled_examples)
        self._save_json(split_examples)
        return labeled_examples

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        inner_text = innertext(xml)
        tokenized = self.tokenizer(
            inner_text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )  # uses Whitespace as pre_processor
        token_level_labels = {}
        for code_map in self.code_maps:
            xml_encoder = XMLEncoder(code_map)
            xml_encoded = xml_encoder.encode(xml)
            labels = self._align_labels(tokenized, xml_encoded, inner_text, code_map)
            token_level_labels[code_map.name] = labels
        return tokenized, token_level_labels

    def _align_labels(self, tokenized: Encoding, xml_encoded: Dict, inner_text, code_map: CodeMap) -> List[int]:
        # prefill with outside of entity label 'O' using IOB2 scheme
        token_level_labels = ['O'] * len(tokenized.input_ids)
        # tokenizer may have truncated the example
        last_token_start, last_token_end = tokenized.offset_mapping[-2]  # -2 because the last token is </s> with offsets (0,0) by convention
        for element_start, element_end in xml_encoded['offsets']:
            # check we are still within the truncated example
            if (element_start <= last_token_start) & (element_end <= last_token_end):
                start_token_idx = tokenized.char_to_token(element_start)
                end_token_idx = tokenized.char_to_token(element_end - 1)  # element_end is the position just after the last token
                assert start_token_idx is not None, f"\n\nproblem with start token for text {inner_text[element_start:element_end]}\n\n{inner_text}\n\n{tokenized.tokens}"
                assert end_token_idx is not None, f"\n\nproblem with end token for text {inner_text[element_start:element_end]}\n\n{inner_text}\n\n{tokenized.tokens}"
                code = xml_encoded['label_ids'][element_start]  # element_end would give the same, maybe check with assert
                assert xml_encoded['label_ids'][element_start] == xml_encoded['label_ids'][element_end - 1], f"{xml_encoded['label_ids'][element_start:element_end]}\n{element_start, element_end}"
                prefix = "B"  # for beginign token according to IOB2 scheme
                label = ''
                for token_idx in range(start_token_idx, end_token_idx + 1):
                    label = self._int_code_to_iob2_label(prefix, code, code_map)
                    token_level_labels[token_idx] = label
                    prefix = "I"  # for subsequet inside tokens
            else:
                # the last token has been reached, no point scanner further elemnts
                break
        return token_level_labels

    def _int_code_to_iob2_label(self, prefix: str, code: int, code_map: CodeMap) -> str:
        label = code_map.constraints[code]['label']
        iob2_label = f"{prefix}-{label}"
        return iob2_label

    def _split(self, examples: List) -> Dict:
        shuffle(examples)
        N = len(examples)
        valid_fraction = min(floor(N * self.split_ratio['eval']), self.split_ratio['max_eval'] - 1)
        test_fraction = min(floor(N * self.split_ratio['test']), self.split_ratio['max_test'] - 1)
        train_fraction = N - valid_fraction - test_fraction
        split_examples = {}
        split_examples['train'] = [e for e in examples[0:train_fraction]]
        split_examples['eval'] = [e for e in examples[train_fraction:train_fraction + valid_fraction]]
        split_examples['test'] = [p for p in examples[train_fraction + valid_fraction:]]
        return split_examples

    def _save_json(self, split_examples: Dict):
        if not self.dest_dir_path.exists():
            self.dest_dir_path.mkdir()
        # basename = "_".join([self.code_map.__name__, now()])
        for subset, examples in split_examples.items():
            # saving line by line to json-line file
            filepath = self.dest_dir_path / f"{subset}.jsonl"
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
    example = "<xml>Here <sd-panel>it is: <i>nested in <sd-tag category='entity' type='gene' role='intervention'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End</sd-panel>."
    example += '_' * 150 + '</xml>'  # to test truncation
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    path = Path('/tmp/test_dataprep')
    path.mkdir()
    source_path = path / 'source'
    source_path.mkdir()
    dest_dir_path = path / 'dataset'
    source_file_path = source_path / 'example.xml'
    source_file_path.write_text(example)
    max_length = 20  # in token!
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
        ]
    }
    expected_tokens = [
        '<s>',
        'Here', 'Ġit', 'Ġis', ':', 'Ġnested', 'Ġin',
        'ĠCre', 'b', '-', '1',
        'Ġwith', 'Ġsome',
        'Ġtail',
        '.', 'ĠEnd', '.',
        '________________________________________________________________',
        '________________________________________________________________',
        '</s>'
    ]
    try:
        data_prep = Preparator(source_path, dest_dir_path, tokenizer, [sd.ENTITY_TYPES, sd.GENEPROD_ROLES], max_length=max_length)
        labeled_examples = data_prep.run()
        print("\nLabel codes: ")
        print(labeled_examples[0]['label_ids'])
        print('\nTokens')
        print(labeled_examples[0]['tokenized'].tokens())
        labeled_example_label_ids = labeled_examples[0]['label_ids']
        assert labeled_example_label_ids['entity_types'] == expected_label_codes['entity_types'], labeled_example_label_ids['entity_types']
        assert labeled_example_label_ids['geneprod_roles'] == expected_label_codes['geneprod_roles'], labeled_example_label_ids['geneprod_roles']
        assert labeled_examples[0]['tokenized'].tokens() == expected_tokens
        assert data_prep.verify()
        filepaths = list(dest_dir_path.glob("*.jsonl"))
        for filepath in filepaths:
            print(f"\nContent of saved file ({filepath}):")
            with filepath.open() as f:
                for line in f:
                    j = json.loads(line)
                    print(json.dumps(j, indent=2))
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
        dest_dir_path = args.dest_dir
        code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES]
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        sdprep = Preparator(Path(source_dir_path), Path(dest_dir_path), tokenizer, code_maps)
        sdprep.run()
        sdprep.verify()
        print("\nDone!")
    else:
        self_test()
