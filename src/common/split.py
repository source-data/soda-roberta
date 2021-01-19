from pathlib import Path
import shutil
import argparse
from math import floor
from random import shuffle
from .utils import progress
from .config import config


def distribute(path: Path, ext: str):
    """Simple utility to split files into train, eval, and test subsets.
    Files with the specified extensions are globed inside a directory and its subdirectories.
    Files will be selected randomly and moved into train/ eval/ and test/ subdirectories.
    If the subdirectories do not exist they are created. If they already exists, the files are redistributed
    The ratios of files moved into each subdirectories is specified in common.config.split_ratio.
    The maximum number of files in eval/ and train/ is specified in common.config.split_ratio as well.

    Args:
        path (Path):
            The path of the directory containing the list of files.
        ext (str):
            Only the files with this extension (WITHOUT THE DOT) will be redistributed.
    """

    print(f"Looking for files with extension {ext} in {path}.")
    filepaths = list(path.glob(f"**/*.{ext}"))
    N = len(filepaths)
    print(f"\nFound {N} files in {path}.\n")
    valid_fraction = min(floor(N * config.split_ratio['eval']), config.split_ratio['max_eval'] - 1)
    test_fraction = min(floor(N * config.split_ratio['test']), config.split_ratio['max_test'] - 1)
    train_fraction = N - valid_fraction - test_fraction
    assert train_fraction + valid_fraction + test_fraction == N
    shuffle(filepaths)
    subset = {}
    subset['train'] = [p for p in filepaths[0:train_fraction]]
    subset['eval'] = [p for p in filepaths[train_fraction:train_fraction + valid_fraction]]
    subset['test'] = [p for p in filepaths[train_fraction + valid_fraction:]]
    for train_valid_test in subset:
        subset_path = path / train_valid_test
        if not subset_path.exists():
            subset_path.mkdir()
        for i, p in enumerate(subset[train_valid_test]):
            progress(i, len(subset[train_valid_test]), f"{train_valid_test} {i+1}                   ")
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
    distribute(Path("/tmp/test_corpus"), "testx")
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
    parser.add_argument('corpus', nargs="?", help='path to the corpus of documents to use.')
    parser.add_argument('-X', '--extension', default='txt', help='Extension (WITHOUT THE DOT) for allowed files in the corpus.')
    args = parser.parse_args()

    if not args.corpus:
        self_test()
    else:
        corpus = args.corpus
        ext = args.extension
        distribute(Path(corpus), ext=ext)


if __name__ == '__main__':
    main()
