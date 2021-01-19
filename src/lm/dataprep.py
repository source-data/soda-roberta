from pathlib import Path
from typing import List, Dict
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from transformers import RobertaTokenizerFast, BatchEncoding
import spacy
import celery
from common.utils import progress
from common import TOKENIZER_PATH
from common.config import config
from . import app
# from .tasks import aligned_tokenization

if config.from_pretrained:
    TOKENIZER = RobertaTokenizerFast.from_pretrained(config.from_pretrained)
else:
    TOKENIZER = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
NLP = spacy.load('en_core_web_sm')


class Preparator:
    """Processes source text documents into examples that can be used in a masked language modeling task.
    It tokenizes the text with the provided tokenizer. 
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_dir_path (Path):
            The path to the source xml files.
        dest_dir_path (Path):
            The path of the destination directory where the files with the encoded labeled examples should be saved.
        ext (str):
            The extension (WITHOUT THE DOT) of the files to be coded.
    """
    def __init__(
        self,
        source_dir_path: Path,
        dest_dir_path: Path,
        max_length: int = config.max_length,
        ext: str = 'txt',
    ):
        self.source_dir_path = source_dir_path
        self.dest_dir_path = dest_dir_path
        self.max_length = max_length
        assert self._dest_dir_is_empty(), f"{self.dest_dir_path} is not empty! Will not overwrite pre-existing dataset."
        if not self.dest_dir_path.exists():
            self.dest_dir_path.mkdir()
        self.filepaths = list(self.source_dir_path.glob(f"**/*.{ext}"))
        shuffle(self.filepaths)

    def _dest_dir_is_empty(self) -> bool:
        if self.dest_dir_path.exists():
            # https://stackoverflow.com/a/57968977
            return not any([True for _ in self.dest_dir_path.iterdir()])
        else:
            return True

    def run(self) -> List:
        """Runs the coding of the examples.
        Saves the resulting text files to the destination directory.
        """
        batch_size = config.celery_batch_size
        N = len(self.filepaths)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            progress(end-1, N, f"{start}:{end} of {N}")
            task_list = [
                aligned_tokenization.s(str(filepath), str(self.dest_dir_path), self.max_length)
                for filepath in self.filepaths[start:end]
            ]
            job = celery.group(task_list)
            results = job.apply_async(
                # retry=True, retry_policy={
                #     'max_retries': 3,
                #     'interval_start': 0,
                #     'interval_step': 0.2,
                #     'interval_max': 0.2,
                # }
            )
            results.get()

    def verify(self):
        filepaths = self.dest_dir_path.glob("**/*.jsonl")
        for p in filepaths:
            with p.open() as f:
                for n, line in enumerate(f):
                    j = json.loads(line)
                    assert len(j['input_ids']) <= self.max_length + 2, f"Length verification: error line {n} in {p} with num_tokens: {len(j['input_ids'])} > {self.max_length + 2}."
                    assert len(j['label_ids']) == len(j['input_ids']), f"mismatch in number of input_ids and label_ids: error line {n} in {p}"
                    assert len(j['special_tokens_mask']) == len(j['input_ids']), f"mismatch in number of input_ids and special_tokens_mask: error line {n} in {p}"
        print("\nLength verification: OK!")
        return True


@app.task
def aligned_tokenization(filepath: str, dest_dir: str, max_length):
    labeled_example = {}
    example = Path(filepath).read_text()
    if example:
        pos_words = NLP(example)
        tokenized: BatchEncoding = TOKENIZER(
            example,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True
        )
        pos_labels = _align_labels(example, pos_words, tokenized)
        labeled_example = {
            'input_ids': tokenized.input_ids,
            'label_ids': pos_labels,
            'special_tokens_mask': tokenized.special_tokens_mask
        }
        _save_json(labeled_example, dest_dir)


def _align_labels(example, pos_words, tokenized: BatchEncoding) -> List[str]:
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
        if not(start == end):  # not a special or empty token
            pos_token[idx] = pos_char[start]
    return pos_token


def _save_json(example: Dict, dest_dir: str):
    # saving line by line to json-line file
    filepath = Path(dest_dir) / "data.jsonl"
    with filepath.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        f.write(f"{json.dumps(example)}\n")


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
    try:
        data_prep = Preparator(source_path, dest_dir_path, max_length=max_length)
        data_prep.run()
        assert data_prep.verify()
        filepaths = list(dest_dir_path.glob("*.jsonl"))
        for filepath in filepaths:
            print(f"\nContent of saved file ({filepath}):")
            with filepath.open() as f:
                for line in f:
                    example = json.loads(line)
                    # print(json.dumps(example))
                    input_ids = example['input_ids']
                    pos_labels = example['label_ids']
                    assert len(input_ids) == len(pos_labels)
                    print('Example:')
                    for i in range(len(input_ids)):
                        print(f"{TOKENIZER.decode(input_ids[i])}\t{pos_labels[i]}")
    finally:
        shutil.rmtree('/tmp/test_dataprep/')
        print("cleaned up and removed /tmp/test_corpus")
    print("Looks like it is working!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Tokenize text and prepares the datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", nargs="?", help="Directory where the source files are located.")
    parser.add_argument("dest_dir", nargs="?", help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir_path = args.source_dir
    if source_dir_path:
        dest_dir_path = args.dest_dir
        dest_dir_path = Path(dest_dir_path)
        if dest_dir_path.exists():
            source_dir_path = Path(source_dir_path)
            for subset in ["train", "eval", "test"]:
                print(f"Preparing: {subset}")
                sdprep = Preparator(source_dir_path / subset, dest_dir_path / subset)
                sdprep.run()
                sdprep.verify()
            print("\nDone!")
        else:
            print(f"{dest_dir_path} does not exist. Cannot proceed.")
    else:
        self_test()
