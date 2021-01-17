
from pathlib import Path
from typing import List, Dict
from math import floor
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from transformers import RobertaTokenizerFast, BatchEncoding
import spacy
import celery
from ..common.utils import progress
from ..common import TOKENIZER_PATH, LM_DATASET
from ..common.config import config
from .celery import app


@app.task
def aligned_tokenization(filepath: str, dest_dir: str, max_length):
    print('Hi')
    labeled_example = {}
    # example = Path(filepath).read_text()
    # if example:
    #     pos_words = nlp(example)
    #     tokenized: BatchEncoding = tokenizer(
    #         example,
    #         max_length=max_length,
    #         truncation=True,
    #         return_offsets_mapping=True,
    #         return_special_tokens_mask=True,
    #         add_special_tokens=True
    #     )
    #     pos_labels = _align_labels(example, pos_words, tokenized)
    #     labeled_example = {
    #         'tokenized': tokenized,
    #         'label_ids': pos_labels
    #     }
    #     _save_json(labeled_example, dest_dir)
    return labeled_example


# @staticmethod
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
            try:
                pos = pos_char[start]
            except Exception:
                import pdb; pdb.set_trace()
            pos_token[idx] = pos
    return pos_token


# @staticmethod
def _save_json(example: Dict, dest_dir: str):
    # saving line by line to json-line file
    filepath = Path(dest_dir) / "data.jsonl"
    with filepath.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        j = {
            'input_ids': example['tokenized'].input_ids,
            'label_ids': example['label_ids'],
            'special_tokens_mask': example['tokenized'].special_tokens_mask
        }
        f.write(f"{json.dumps(j)}\n")
