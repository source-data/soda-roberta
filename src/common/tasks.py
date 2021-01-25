from typing import List, Dict
import json
from lxml.etree import Element, tostring, parse
from pathlib import Path
from nltk import PunktSentenceTokenizer
from .celery import app
from .config import config
from .utils import innertext, cleanup


@app.task
def align_labels_task(*args, **kwargs):
    return _align_labels(*args, **kwargs)


@app.task
def examples_from_file_task(filepath: str, xpath: str, punkt: bool, keep_xml: bool, remove_tail: bool) -> Dict:
    examples = []
    elements = _parse_xml_file(filepath, xpath, remove_tail)
    examples = _extract_text_from_elements(elements, punkt, keep_xml)
    examples = _cleanup(examples)
    return {'examples': examples, 'filepath': str(filepath)}


def _parse_xml_file(filepath: str, xpath: str, remove_tail: bool) -> List[str]:
    filepath = Path(filepath)
    with filepath.open() as f:
        xml = parse(f)
        elements = xml.xpath(xpath)
        if remove_tail:
            for e in elements:
                if e.tail is not None:
                    e.tail = None
    return elements


def _extract_text_from_elements(elements: Element, punkt: bool, keep_xml: bool) -> List[str]:
    examples = []
    if keep_xml:
        for e in elements:
            xml_str = tostring(e).decode('utf-8')  # tostring returns bytes
            length = len(innertext(e))
            if length > config.min_char_length:
                examples.append(xml_str)
    else:
        for e in elements:
            text = innertext(e)
            if punkt:
                sentences = PunktSentenceTokenizer().tokenize(text=text)
                filtered_sentences = [s for s in sentences if self._filter(s)]
                examples += filtered_sentences
            else:
                if _filter(text):
                    examples.append(text)
    return examples


def _cleanup(examples: List[str]) -> List[str]:
    examples = [cleanup(e) for e in examples]
    return examples


def _filter(example: str) -> str:
    example = example if len(example) > config.min_char_length else ''
    return example


@app.task
def save_task(text: str, dest_dir: Path, basename: str, suffix: str, ext: str):
    ex_filename = f"{basename}_{suffix}.{ext}"
    saving_path = Path(dest_dir) / ex_filename
    if saving_path.exists():
        print(f"{saving_path} already exists. Not overwritten.                                                     ", end="\r", flush=True)
        return 0
    else:
        saving_path.write_text(text)
        return 1


@app.task
def aligned_tokenization_task(filepath: str, dest_dir: str, max_length):
    labeled_example = {}
    example = Path(filepath).read_text()
    if example:
        pos_words = config.nlp(example)
        tokenized = config.tokenizer(
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


def _align_labels(example, pos_words, tokenized) -> List[str]:
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
