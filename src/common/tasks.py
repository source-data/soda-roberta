from typing import List, Dict
import json
from lxml.etree import Element, tostring, parse
from pathlib import Path
from nltk import PunktSentenceTokenizer
from .celery import app
from .config import config
from .utils import innertext, cleanup


@app.task
def examples_from_file_task(filepath: str, xpath: str, punkt: bool, keep_xml: bool, remove_tail: bool) -> Dict:
    """Generates text or xml examples from xml documents. 
    Examples to be extracted are found using an XPath expression.
    THe resulting text can be segmented in sentences if desired. 
    Either the inner text is extracted or the xml of the extracted element is kept.

    Args:
        filepath (str): the path to the source file.
        xpath (str): the XPath expression to identify the example(s) in the xml file.
        punkt (bool): whether to split the text into individual sentences.
        keep_xml (bool): whether to keep the xml of the element, otherwise the inner text is extracted.
        remove_tail (bool): set this to False if the text after the element should be included.
    """
    examples = []
    elements = _parse_xml_file(filepath, xpath, remove_tail)
    examples = _extract_text_from_elements(elements, punkt, keep_xml)
    examples = _cleanup(examples)
    return examples


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
def save_task(text: str, filepath: str,):
    """Writes each text on 1 line at the end of the file.
    Strips text from newlines so that it can be written on a single line.

    Args:
        text (str): the text, will be stripped of newline
        filepath (Path): the path to the file
    """
    with Path(filepath).open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        f.write(f"{text.strip()}\n")
    return 1


@app.task
def aligned_tokenization_task(example: str, dest_file_path: str, max_length):
    labeled_example = {}
    # invoke Spacy's part-of-speech tagger
    # will assign universal POS tags https://universaldependencies.org/u/pos/
    # https://spacy.io/usage/linguistic-features#pos-tagging
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
    _save_json(labeled_example, dest_file_path)


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


def _save_json(example: Dict, dest_file_path: str):
    # saving line by line to json-line file
    with Path(dest_file_path).open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        f.write(f"{json.dumps(example)}\n")
