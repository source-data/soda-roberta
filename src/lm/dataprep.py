from pathlib import Path
from typing import List
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
import celery
from common.utils import progress
from common.config import config
# from common.celery import app
from common.tasks import aligned_tokenization_task


class Preparator:
    """Processes source text documents into examples that can be used in a masked language modeling task.
    It tokenizes the text with the provided tokenizer. 
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_file_path (Path):
            The path to the source file with one example per line.
        dest_dir_path (Path):
            The path of the destination directory where the files with the encoded labeled examples should be saved.
    """
    def __init__(
        self,
        source_file_path: Path,
        dest_file_path: Path,
        max_length: int = config.max_length
    ):
        self.source_file_path = source_file_path
        self.dest_file_path = dest_file_path
        self.max_length = max_length
        assert not self.dest_file_path.exists(), f"{self.dest_file_path} already exists! Will not overwrite pre-existing dataset."

    def run(self) -> List:
        """Runs the coding of the examples.
        Saves the resulting text files to the destination directory.
        """
        batch_size = config.celery_batch_size
        with self.source_file_path.open() as f:
            task_list = []
            for n, line in enumerate(f):
                line = line.strip()
                if line:
                    task_list.append(aligned_tokenization_task.s(line, str(self.dest_file_path), self.max_length))
                if n % batch_size == 0:
                    print(f"{['.   ',' .  ', '  . ', '   .'][(n // batch_size) % 4]} {n}", end="\r", flush=True)
                    job = celery.group(task_list)
                    results = job.apply_async()
                    results.get()
                    task_list = []
            job = celery.group(task_list)
            results = job.apply_async()
            results.get()

    def verify(self):
        with self.dest_file_path.open() as f:
            cumul_len = 0
            max_len = 0
            longest_example = ''
            min_len = 1E3
            shortest_example = ''
            for n, line in enumerate(f):
                j = json.loads(line)
                L = len(j['input_ids'])
                assert L <= self.max_length + 2, f"Length verification: error line {n} in {p} with num_tokens: {len(j['input_ids'])} > {self.max_length + 2}."
                assert L == len(j['input_ids']), f"mismatch in number of input_ids and label_ids: error line {n} in {p}"
                assert L == len(j['special_tokens_mask']), f"mismatch in number of input_ids and special_tokens_mask: error line {n} in {p}"
                cumul_len += L
                if L > max_len:
                    max_len = L
                    longest_example = j['input_ids']
                if L < min_len:
                    min_len = L
                    shortest_example = j['input_ids']
        print("\nLength verification: OK!")
        print(f"\naverage input_ids length = {round(cumul_len / n)} (min={min_len}, max={max_len}) tokens")
        print(f"longest example: {config.tokenizer.decode(longest_example)}")
        print(f"shortest example: {config.tokenizer.decode(shortest_example)}")
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
                        print(f"{config.tokenizer.decode(input_ids[i])}\t{pos_labels[i]}")
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
        source_dir_path = Path(source_dir_path)
        dest_dir_path = args.dest_dir
        if dest_dir_path:
            dest_dir_path = Path(dest_dir_path)
            if not dest_dir_path.exists():
                dest_dir_path.mkdir()
                print(f"{dest_dir_path} created")
            for subset in ["train", "eval", "test"]:
                print(f"Preparing: {subset}")
                sdprep = Preparator(source_dir_path / f"{subset}.txt", dest_dir_path / f"{subset}.jsonl")
                sdprep.run()
                sdprep.verify()
            print("\nDone!")
        else:
            raise ValueError("Please explicitly provide the detination path. Cannot proceed without it.")
    else:
        self_test()
