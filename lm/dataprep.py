from pathlib import Path
from typing import List, Dict
from math import floor
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from transformers import RobertaTokenizerFast, BatchEncoding
import spacy
from common.utils import progress
from common import TOKENIZER_PATH, NER_DATASET
from common.config import config


class Preparator:
    """Processes source text documents into examples that can be used in a masked language modeling task.
    It tokenizes the text with the provided tokenizer. 
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_dir_path (Path):
            The path to the source xml files.
        dest_dir_path (Path):
            The path of the destination directory where the files with the encoded labeled examples should be saved.
        tokenizer (ByteLevelBPETokenizer):
            The pre-trained tokenizer to use for processing the inner text.
    """
    def __init__(
        self,
        source_dir_path: Path,
        dest_dir_path: Path,
        tokenizer: RobertaTokenizerFast,
        max_length: int = config.max_length
    ):
        self.source_dir_path = source_dir_path
        self.dest_dir_path = dest_dir_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.nlp = spacy.load('en_core_web_sm')
        assert self._dest_dir_is_empty(), f"{self.dest_dir_path} is not empty! Will not overwrite pre-existing dataset."

    def _dest_dir_is_empty(self) -> bool:
        if self.dest_dir_path.exists():
            # https://stackoverflow.com/a/57968977
            return not any([True for _ in self.dest_dir_path.iterdir()])
        else:
            return True

    def run(self, ext: str = 'txt') -> List:
        """Runs the coding of the examples.
        Saves the resulting text files to the destination directory.

        Args:
            ext (str):
               The extension (WITHOUT THE DOT) of the files to be coded.
        """
        labeled_examples = []
        filepaths = list(self.source_dir_path.glob(f"**/*.{ext}"))
        for i, filepath in enumerate(filepaths):
            progress(i, len(filepaths), f"{filepath.name}                 ")
            example = filepath.read_text()
            pos_words = self.nlp(example)
            tokenized: BatchEncoding = self.tokenizer(
                example,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=True
            )
            pos_labels = self._align_labels(example, pos_words, tokenized)
            labeled_examples.append({
                'tokenized': tokenized,
                'label_ids': pos_labels
            })
        self._save_json(labeled_examples)
        return labeled_examples

    def _align_labels(self, example, pos_words, tokenized: BatchEncoding) -> List[str]:
        # since spacy and the pre-tokenizer may not split text the same way, we bridge both via a character-level POS tag list
        pos_char = [''] * len(example)  # pos for part-of-speech
        # make a character-level pos from pos_word
        for w in pos_words:
            start = w.idx
            end = start + len(w)
            pos_char[start:end] = [w.pos_] * len(w)
        # convert the character-level POS tags into token-level POS labels
        pos_token = ['X'] * len(tokenized.tokens())  # includes special tokens
        for idx, (start, end) in enumerate(tokenized.offset_mapping):
            if not(start == end == 0):  # not a special token
                pos = pos_char[start]
                pos_token[idx] = pos
        return pos_token

    def _save_json(self, examples: Dict):
        if not self.dest_dir_path.exists():
            self.dest_dir_path.mkdir()
        # saving line by line to json-line file
        filepath = self.dest_dir_path / "data.jsonl"
        with filepath.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
            shuffle(examples)
            for example in examples:
                j = {
                    'input_ids': example['tokenized'].input_ids,
                    'label_ids': example['label_ids']
                }
                f.write(f"{json.dumps(j)}\n")

    def verify(self):
        filepaths = self.dest_dir_path.glob("**/*.jsonl")
        for p in filepaths:
            with p.open() as f:
                for n, line in enumerate(f):
                    j = json.loads(line)
                    assert len(j['input_ids']) <= self.max_length + 2, f"Length verification: error line {n} in {p} with num_tokens: {len(j['input_ids'])} > {self.max_length + 2}."
                    assert len(j['label_ids']) == len(j['input_ids']), f"mismatch in number of input_ids and label_ids: error line {n} in {p}"
        print("\nLength verification: OK!")
        return True


def self_test():
    example = "Here it is: nested in Creb-1 with some tail. The end."
    example += ' 1 2 3 4 5 6 7 8 9 0'  # to test truncation
    path = Path('/tmp/test_dataprep')
    path.mkdir()
    source_path = path / 'source'
    source_path.mkdir()
    dest_dir_path = path / 'dataset'
    source_file_path = source_path / 'example.txt'
    source_file_path.write_text(example)
    max_length = 20  # in token!
    expected_tokens = [
        '<s>', 'Here', 'Ġit', 'Ġis', ':', 'Ġnested', 'Ġin', 'ĠCre', 'b', '-', '1', 'Ġwith', 'Ġsome', 'Ġtail', '.', 'ĠThe', 'Ġend', '.', 'Ġ1', '</s>'
    ]
    expected_labels = [
        'X', 'ADV', 'PRON', 'AUX', 'PUNCT', 'VERB', 'ADP', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'ADP', 'DET', 'NOUN', 'PUNCT', 'DET', 'NOUN', 'PUNCT', 'NUM', 'X'
    ]
    try:
        data_prep = Preparator(source_path, dest_dir_path, tokenizer, max_length=max_length)
        examples = data_prep.run()
        tokens = examples[0]['tokenized'].tokens()
        input_ids = examples[0]['tokenized'].input_ids
        pos_labels = examples[0]['label_ids']
        print(pos_labels)
        print('\nTokens')
        for i in range(len(tokens)):
            print(f"{tokens[i]}\t{tokenizer.decode(input_ids[i])}\t{pos_labels[i]}")
        assert tokens == expected_tokens
        assert pos_labels == expected_labels
        assert data_prep.verify()
        filepaths = list(dest_dir_path.glob("*.jsonl"))
        for filepath in filepaths:
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
    parser = ArgumentParser(description="Tokenize text and prepares the datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", nargs="?", help="Directory where the source files are located.")
    parser.add_argument("dest_dir", nargs="?", default=NER_DATASET, help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir_path = args.source_dir
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    if source_dir_path:
        dest_dir_path = args.dest_dir
        dest_dir_path = Path(dest_dir_path)
        source_dir_path = Path(source_dir_path)
        for subset in ["train", "eval", "test"]:
            print(f"Preparing: {subset}")
            sdprep = Preparator(source_dir_path / subset, dest_dir_path / subset, tokenizer)
            sdprep.run()
            sdprep.verify()
        print("\nDone!")
    else:
        self_test()
