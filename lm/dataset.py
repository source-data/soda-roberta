from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from common.utils import progress
from common.config import config
from common import LM_DATASET, TOKENIZER_PATH


class BioDataset(Dataset):
    def __init__(self, path: Path, tokenizer: RobertaTokenizerFast, subset: str):

        self.examples = []
        path = path / subset
        src_files = list(path.glob("*.txt"))
        for i, src_file in enumerate(src_files):
            progress(i, len(src_files), f"🚀 {src_file.name}                  ")
            text = src_file.read_text(encoding="utf-8")
            encoded = tokenizer(
                text,
                truncation=True,
                add_special_tokens=True
            )
            self.examples.append(encoded.input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


def self_test():
    max_len = config.max_length
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=max_len)
    dataset = BioDataset(Path(LM_DATASET), tokenizer, "test")
    assert len(dataset) > 0
    for i, e in enumerate(dataset.examples):
        progress(i, len(dataset), f"len({i}) <= {max_len + 2}                                ")
        assert len(e) <= max_len + 2, f"{len(e)} != {max_len + 2}" # +2 because of </s> and <s> token
    print("\nLooks OK!")


if __name__ == '__main__':
    self_test()
