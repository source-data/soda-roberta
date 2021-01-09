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
    The XML is used to generate labels according to the provided CodeMaps.
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
            labeled_example = self._encode_example(xml_example)
            labeled_examples.append(labeled_example)
        split_examples = self._split(labeled_examples)
        self._save_json(split_examples)
        return labeled_examples

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml, self.tokenizer)
        token_level_labels = {}
        for code_map in self.code_maps:
            xml_encoded = xml_encoder.encode(
                code_map,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )
            input_ids = xml_encoded['token_input_ids']
            token_level_labels[code_map.name] = xml_encoded['token_labels']
        return {
            'input_ids': input_ids,
            'label_ids': token_level_labels
        }

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
                    f.write(f"{json.dumps(example)}\n")

    def verify(self):
        filepaths = self.dest_dir_path.glob("**/*.jsonl")
        for p in filepaths:
            with p.open() as f:
                for n, line in enumerate(f):
                    j = json.loads(line)
                    L = len(j['input_ids'])
                    assert L <= self.max_length, f"Length verification: error line {n} in {p} with {len(j['tokens'])} tokens > {self.max_length}."
                    for k, label_ids in j['label_ids'].items():
                        assert len(label_ids) == L, f"mismatch in number of tokens and {k} label_ids: error line {n} in {p}"
        print("\nLength verification: OK!")
        return True


def self_test():
    # example = "<xml><a>  </a>.</xml>"
    example = "<xml>Here <sd-panel>it is<sd-tag role='reporter'> </sd-tag>: <i>nested <sd-tag role='reporter'>in</sd-tag> <sd-tag category='entity' type='gene' role='intervention'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End </sd-panel>."
    example += ' 1 2 3 4 5 6 7 8 9 0' + '</xml>'  # to test truncation
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    path = Path('/tmp/test_dataprep')
    path.mkdir()
    source_path = path / 'source'
    source_path.mkdir()
    dest_dir_path = path / 'dataset'
    source_file_path = source_path / 'example.xml'
    source_file_path.write_text(example)
    max_length = 22  # in token!
    expected_tokens = [
        '<s>', 'Here', 'Ġ', 'it', 'Ġis', 'Ġ', ':', 'Ġ', 'n', 'ested', 'Ġ', 'in', 'Ġ', 'Cre', 'b', '-', '1', 'Ġwith', 'Ġsome', 'Ġ', 'tail', '</s>'
    ]
    expected_label_codes = {
        'entity_types': [
            # '<s>', 'Here', 'Ġ', 'it', 'Ġis', 'Ġ', ':', 'Ġ', 'n', 'ested', 'Ġ', 'in', 'Ġ', 'Cre',        'b',          '-',          '1',          'Ġwith', 'Ġsome', 'Ġ', 'tail',       '</s>'
              'O',   'O',    'O', 'O',  'O',   'O', 'O' ,'O', 'O', 'O',     'O',  'O', 'O', 'B-GENEPROD', 'I-GENEPROD', 'I-GENEPROD', 'I-GENEPROD', 'O',     'O',     'O', 'B-GENEPROD', 'O'
        ],
        'geneprod_roles': [
              'O',  'O',     'O', 'O',  'O',   'O', 'O', 'O', 'O', 'O',     'O',  'O', 'O', 'B-CONTROLLED_VAR', 'I-CONTROLLED_VAR', 'I-CONTROLLED_VAR', 'I-CONTROLLED_VAR', 'O', 'O', 'O', 'B-MEASURED_VAR', 'O'
        ],
        'panel_start': [
              'O', 'O',      'O', 'B-PANEL_START', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
        ]
    }
    try:
        data_prep = Preparator(source_path, dest_dir_path, tokenizer, [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.PANELIZATION], max_length=max_length)
        labeled_examples = data_prep.run()
        print("\nXML examples:")
        print(example)
        print('\nTokens')
        tokens = [tokenizer.convert_ids_to_tokens(id) for id in labeled_examples[0]['input_ids']]
        print(tokens)
        print("\nLabel codes: ")
        print(labeled_examples[0]['label_ids'])

        labeled_example_label_ids = labeled_examples[0]['label_ids']
        assert tokens == expected_tokens, tokens
        assert labeled_example_label_ids['entity_types'] == expected_label_codes['entity_types'], labeled_example_label_ids['entity_types']
        assert labeled_example_label_ids['geneprod_roles'] == expected_label_codes['geneprod_roles'], labeled_example_label_ids['geneprod_roles']
        assert labeled_example_label_ids['panel_start'] == expected_label_codes['panel_start'], labeled_example_label_ids['panel_start']
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
        code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.BORING, sd.PANELIZATION]
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        sdprep = Preparator(Path(source_dir_path), Path(dest_dir_path), tokenizer, code_maps)
        sdprep.run()
        sdprep.verify()
        print("\nDone!")
    else:
        self_test()
