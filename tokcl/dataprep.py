from pathlib import Path
from typing import List, Tuple
from xml.etree.ElementTree import parse, Element, ElementTree
import shutil
from tokenizers import Encoding, ByteLevelBPETokenizer
from .encoder import XMLEncoder
from .xmlcode import (
    CodeMap, EntityTypeCodeMap
)
from common.utils import innertext
from common import TOKENIZER_PATH


class SDPreparator:
    """Processes source xml documents into examples that can be used in a token classification task.

    Args:
        source_dir_path (Path):
            The path to the source xml files.
        dest_dir_path (Path):
            The path to the desitnation directory where the encoded labeled examples should be saved.
        tokenizer (ByteLevelBPETokenizer):
            The pre-trained tokenizer to use for processing the inner text.
        code_map (CodeMap):
            The XML-to-code constraints mapping label codes to specific combinations of tag name and attribute values.
    """
    def __init__(self, source_dir_path: Path, dest_dir_path: Path, tokenizer: ByteLevelBPETokenizer, code_map: CodeMap):
        self.source_dir_path = source_dir_path
        self.dest_dir_path = dest_dir_path
        self.ec = XMLEncoder(code_map)
        self.tokenizer = tokenizer

    def run(self, ext: str = 'xml'):
        """Runs the coding and labeling of xml examples. 
        Saves the resulting text files to the destination directory.

        Args:
            ext (str):
               The extension (WITHOUT THE DOT) of the files to be coded.
        """
        labeled_examples = []
        filepaths = self.source_dir_path.glob(f"**/*.{ext}")
        for filepath in filepaths:
            with filepath.open() as f:
                xml_example: ElementTree = parse(f)
            xml_example: Element = xml_example.getroot()
            labeled_token = self._encode_example(xml_example)
            labeled_examples.append(labeled_token)
            self._save(filepath, labeled_token)
        return labeled_examples

    def _encode_example(self, xml: Element) -> List:
        character_level_label_ids = self.ec.encode(xml)
        character_level_label_ids = [c if c is not None else -100 for c in character_level_label_ids]
        inner_text = innertext(xml)
        tokenized = self.tokenizer.encode(inner_text)  # uses Whitespace as pre_processor
        labeled_token = self._align_labels(tokenized, character_level_label_ids)
        return labeled_token

    def _align_labels(self, tokenized: Encoding, character_level_label_ids: list) -> Tuple[List[int], List[str], List[int]]:
        token_level_label_ids = []
        for i in range(len(tokenized.tokens)):
            start, end = tokenized.offsets[i]
            label_id = character_level_label_ids[end - 1]  # better to take last character of token since firt can be special character for space; but see # https://huggingface.co/transformers/custom_datasets.html#tok-ner
            token_level_label_ids.append(label_id)
        return (token_level_label_ids, tokenized.tokens, tokenized.ids)

    def _save(self, source_filepath: Path, labeled_token: Tuple[List[int], List[str], List[int]]):
        basename = source_filepath.stem
        labels = labeled_token[0]
        tokens = labeled_token[1]
        token_ids = labeled_token[2]
        path = self.dest_dir_path / f"{basename}.txt"
        with path.open('w') as f:
            for i in range(len(tokens)):
                line = "\t".join([
                    tokens[i],
                    str(token_ids[i]),
                    str(labels[i]),
                ])
                f.write(f"{line}\n")


def self_test():
    example = "<xml>Here <sd-panel>it is: <i>nested in <sd-tag category='entity' type='gene' role='assayed'>Creb-1</sd-tag> with some <sd-tag type='cell'>tail</sd-tag></i>. End</sd-panel>.</xml>"
    tokenizer = ByteLevelBPETokenizer.from_file(
        f"{TOKENIZER_PATH}/vocab.json",
        f"{TOKENIZER_PATH}/merges.txt"
    )
    path = Path('/tmp/test_dataprep')
    path.mkdir()
    source_path = path / 'source'
    source_path.mkdir()
    dest_path = path / 'dest'
    dest_path.mkdir()
    source_file_path = source_path / 'example.xml'
    source_file_path.write_text(example)
    expected_label_codes = [-100, -100, -100, -100, -100, -100, 2, 2, 2, 2, -100, -100, 4, -100, -100, -100]
    expected_tokens = ['Here', 'Ġit', 'Ġis', ':', 'Ġnested', 'Ġin', 'ĠCre', 'b', '-', '1', 'Ġwith', 'Ġsome', 'Ġtail', '.', 'ĠEnd', '.']
    try:
        data_prep = SDPreparator(source_path, dest_path, tokenizer, EntityTypeCodeMap)
        labeled_examples = data_prep.run()
        print("\nLabel codes: ")
        print(labeled_examples[0][0])
        assert labeled_examples[0][0] == expected_label_codes
        print('\nTokens')
        print(labeled_examples[0][1])
        assert labeled_examples[0][1] == expected_tokens
        dest_file_path = dest_path / 'example.txt'
        print(f"\nContent of saved file ({dest_file_path}):")
        print(dest_file_path.read_text())
    finally:
        shutil.rmtree('/tmp/test_dataprep/')
        print("cleaned up and removed /tmp/test_corpus")
    print(f"Looks like it is working!")


if __name__ == "__main__":
    self_test()
