from pathlib import Path
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
