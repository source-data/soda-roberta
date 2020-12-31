from pathlib import Path
import shutil
import argparse
from math import floor
from random import shuffle
from ..common.utils import progress
from ..common.config import config
from ..common import DATASET


def distribute(path: Path, allowed_extension: str = '.nxml'):
    """Simple utility to split a set of files into train, eval, and test subsets.
    Files will be selected randomly and moved into train/ eval/ and test/ subdirectories.
    The ratios of files moved into each subdirectories is specified in common.config.split_ratio

    Args:
        path (Path):
            The path of the directory containing the list of files.
        allowed_extension (str):
            Only the files with this extension (WITH THE DOT) will be redistributed.
    """

    filepaths = [f for f in path.iterdir() if f.suffix == allowed_extension]
    N = len(filepaths)
    train_fraction = floor(N * config.split_ratio['train'])
    valid_fraction = floor(N * config.split_ratio['eval'])
    shuffle(filepaths)
    subset = {}
    subset['train'] = [p for p in filepaths[0:train_fraction]]
    subset['eval'] = [p for p in filepaths[train_fraction:train_fraction + valid_fraction]]
    subset['test'] = [p for p in filepaths[train_fraction + valid_fraction:]]
    for train_valid_test in subset:
        subset_path = path / train_valid_test
        subset_path.mkdir()
        for i, p in enumerate(subset[train_valid_test]):
            progress(i, len(subset[train_valid_test]), train_valid_test)
            filename = p.name
            p.rename(subset_path / filename)
        print()


def self_test():
    """Call module to self-test it.
    """
    Path('/tmp/test_corpus').mkdir()
    expected = []
    N = 10
    for i in range(N):
        filename = f"test_file_{i}.testx"
        expected.append(filename)
        p = Path("/tmp/test_corpus") / filename
        p.write_text(f"test {i}")
    distribute(Path("/tmp/test_corpus"), ".testx")
    subsets = list(Path("/tmp/test_corpus").iterdir())
    subsets_end = [s.parts[-1] for s in subsets]
    try:
        assert set(subsets_end) == set(["train", "eval", "test"]), subsets_end
        n = 0
        for subset in subsets:
            for p in subset.iterdir():
                assert p.name in expected, f"{p.name} should not be there"
                n += 1
        assert n == N, f"only {n} out of {N} files found"
        print("It seems to work!")
    finally:
        shutil.rmtree('/tmp/test_corpus/')
        print("cleaned up and removed /tmp/test_corpus")


def main():
    parser = argparse.ArgumentParser(description='Splitting a corpus into train, valid and testsets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', nargs="?", default=DATASET, help='path to the corpus of documents to use.')
    parser.add_argument('-X', '--extension', default='.txt', help='Extension for allowed files in the corpus.')
    args = parser.parse_args()

    if not args.corpus:
        self_test()
    else:
        corpus = args.corpus
        ext = args.extension
        distribute(Path(corpus), allowed_extension=ext)


if __name__ == '__main__':
    main()
