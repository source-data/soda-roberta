from pathlib import Path
from typing import List, Tuple, Dict
import json
from lxml.etree import fromstring, Element
from random import shuffle
import celery
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding, ByT5Tokenizer
from .xml2labels import CodeMap
from .encoder import XMLEncoder
from .celery import app
from .config import config
from .utils import innertext


# Celery tasks
# @app.task
def aligned_tokenization_task(example: str, dest_file_path: str, max_length: List[int]):
    labeled_example = {}
    # character-level ByT5Tokenizer tokenizer converts string into bytes
    # for unicode character several bytes will be used
    # This leads to shifts as compared to spacy nlp part of speech tagging
    # so we replace anything non ascii by ?
    if isinstance(config.tokenizer, ByT5Tokenizer):
        example = str(example.encode('ascii', 'replace'))
    # invoke Spacy's part-of-speech tagger
    # will assign universal POS tags https://universaldependencies.org/u/pos/
    # https://spacy.io/usage/linguistic-features#pos-tagging
    pos_words = config.nlp(example)
    examples = example.split(config.twin_delimiter)
    tokenized_examples = []
    pos_labels = []
    for ex, max_l in zip(examples, max_length):
        if config.tokenizer.is_fast:
            tokenized = config.tokenizer(
                ex,
                max_length=max_l,
                truncation=config.truncation,
                return_offsets_mapping=True,
                return_special_tokens_mask=True
            )
        else:
            tokenized = config.tokenizer(
                ex,
                max_length=max_l,
                truncation=config.truncation,
                # python tokenizers do not have return_offsets_mapping
                return_special_tokens_mask=True
            )
            # calculate ourselves offsets mapping
            offsets_mapping = _get_offset_mapping(tokenized, config.tokenizer)
            tokenized['offset_mapping'] = offsets_mapping
        tokenized_examples.append(tokenized)
        pos_labels.append(_align_labels(ex, pos_words, tokenized))
    if len(tokenized_examples) == 1:
        assert len(pos_labels) == 1
        input_ids = tokenized_examples[0]["input_ids"]
        special_tokens_mask = tokenized_examples[0]["special_tokens_mask"]
        pos_labels = pos_labels[0]
    else:
        input_ids = [t["input_ids"] for t in tokenized_examples]
        special_tokens_mask = [t["special_tokens_mask"] for t in tokenized_examples]
    labeled_example = {
        'max_length': max_length,
        'input_ids': input_ids,
        'label_ids': pos_labels,
        'special_tokens_mask': special_tokens_mask
    }
    _save_json(labeled_example, dest_file_path)


if config.asynchr:
    aligned_tokenization_task = app.task(aligned_tokenization_task)


def _get_offset_mapping(tokenized, tokenizer):
    start = 0
    offset_mapping = []
    for input_id in tokenized.input_ids:
        token = tokenizer.convert_ids_to_tokens([input_id], skip_special_tokens=True)  # the list [inputs_id] is a trick to allows skipping a special token and returning an empty list
        # test if token is special token in which case offset is (0, 0) by convention
        if token:
            length = len(token[0])
            offset_mapping.append((start, start+length))
            start += length
        else:
            offset_mapping.append((0, 0))
    return offset_mapping


def _align_labels(example: str, pos_words, tokenized: BatchEncoding) -> List[str]:
    # since spacy and the pre-tokenizer may not split text the same way, we bridge both via a character-level POS tag list
    pos_char = [''] * len(example)  # pos for part-of-speech
    # make a character-level pos from pos_word
    for w in pos_words:
        start = w.idx
        end = start + len(w)
        pos_char[start:end] = [w.pos_] * len(w)
    # convert the character-level POS tags into token-level POS labels

    pos_token = ['X'] * len(tokenized.input_ids)  # includes special tokens
    for idx, (start, end) in enumerate(tokenized.offset_mapping):
        if not(start == end):  # not a special or empty token
            try:
                pos_token[idx] = pos_char[start]
            except IndexError:
                raise ValueError(f"{'|'.join([config.tokenizer.convert_ids_to_tokens(i) for i in tokenized.input_ids])}\n{str(pos_words)}")
    return pos_token


def _save_json(example: Dict, dest_file_path: str):
    # saving line by line to json-line file
    with Path(dest_file_path).open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        f.write(f"{json.dumps(example)}\n")


def _special_tokens_mask(tokens, tokenizer):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    unk_token = tokenizer.unk_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    cls_token = tokenizer.cls_token
    mask_token = tokenizer.mask_token
    special_tokens = set([bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token])
    special_tokens_mask = [1 if t in special_tokens else 0 for t in tokens]
    return special_tokens_mask


class PreparatorLM:
    """Processes source text documents into examples that can be used in a masked language modeling task.
    It tokenizes the text with the provided tokenizer.
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_dir_path (Path):
            The path to the directory of the source files with one example per line.
        dest_dir_path (Path):
            The path of the destination directory where the files with the encoded labeled examples should be saved.
    """
    def __init__(
        self,
        source_dir_path: str,
        dest_dir_path: str,
        max_length: int = config.max_length,
        subsets: List[str] = ["train", "eval", "test"]
    ):
        self.source_dir_path = Path(source_dir_path)
        self.dest_dir_path = dest_dir_path
        self.subsets = subsets
        if not self.dest_dir_path:
            basename = self.source_dir_path.name
            self.dest_dir_path = Path("/data/json") / basename
        else:
            self.dest_dir_path = Path(self.dest_dir_path)
        if self.dest_dir_path.exists():
            raise ValueError(f"{self.dest_dir_path} already exists! Will not overwrite pre-existing dataset.")
        elif not self.dest_dir_path.parents[0].exists():
            raise ValueError(f"{self.dest_dir_path.parents[0]} does not exist, cannot proceed")
        else:
            self.dest_dir_path.mkdir()
            print(f"{self.dest_dir_path} created")
        self.max_length = max_length

    def run(self) -> List:
        """Runs the coding of the examples.
        Saves the resulting text files to the destination directory.
        """
        for subset in self.subsets:
            print(f"Preparing: {subset}")
            source_file_path = self.source_dir_path / f"{subset}.txt"
            dest_file_path = self.dest_dir_path / f"{subset}.jsonl"
            batch_size = config.celery_batch_size
            max_length = config.max_length
            if isinstance(max_length, int):
                max_length = [max_length]
            # n = len(max_length)
            with source_file_path.open() as f:
                task_list = []
                lines = f.readlines()
                i = 0
                for line in tqdm(lines):
                    line = line.strip()
                    # cycling across max length
                    # necessary to truncate appropriately twin examples datasets
                    # for example titles are shorter than abstracts
                    # max_l = max_length[i % n]
                    if line:
                        if config.asynchr:
                            task_list.append(aligned_tokenization_task.s(line, str(dest_file_path), max_length))
                        else:
                            aligned_tokenization_task(line, str(dest_file_path), max_length)
                    if config.asynchr:
                        if i % batch_size == 0:
                            job = celery.group(task_list)
                            results = job.apply_async()
                            results.get()
                            task_list = []
                    i += 1
                if config.asynchr:
                    job = celery.group(task_list)
                    results = job.apply_async()
                    results.get()
                    # ORDER OF RESULTS IS NOT GUARANTEED!
            # self._verify(dest_file_path)

    def _verify(self, dest_file_path: Path):
        with dest_file_path.open() as f:
            cumul_len = 0
            max_len = 0
            longest_example = ''
            min_len = 1E3
            shortest_example = ''
            max_length = config.max_length
            if isinstance(max_length, int):
                max_length = [max_length]
            n = len(max_length)
            for i, line in enumerate(f):
                j = json.loads(line)
                L = len(j['input_ids'])
                max_l = max_length[i % n]  # cycle across max length, necessary for twin examples datasets
                assert L <= max_l + 2, f"Length verification: error line {i} in {str(dest_file_path)} with num_tokens: {len(j['input_ids'])} > {max_l + 2}."
                assert L == len(j['input_ids']), f"mismatch in number of input_ids and label_ids: error line {i} in {str(dest_file_path)}"
                assert L == len(j['special_tokens_mask']), f"mismatch in number of input_ids and special_tokens_mask: error line {i} in {str(dest_file_path)}"
                cumul_len += L
                if L > max_len:
                    max_len = L
                    longest_example = j['input_ids']
                if L < min_len:
                    min_len = L
                    shortest_example = j['input_ids']
        i += 1
        print("\nLength verification: OK!")
        print(f"\naverage input_ids length = {round(cumul_len / i)} (min={min_len}, max={max_len}) tokens")
        print(f"longest example: {config.tokenizer.decode(longest_example)}")
        print(f"shortest example: {config.tokenizer.decode(shortest_example)}")
        return True


class PreparatorTOKCL:
    """Processes source xml documents into examples that can be used in a token classification task.
    It tokenizes the text with the provided tokenizer.
    The XML is used to generate labels according to the provided CodeMap.
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_file_path (Path):
            The path to the source file.
        dest_file_path (Path):
            The path of the destination file where the files with the encoded labeled examples should be saved.
        tokenizer (RobertaTokenizerFast):
            The pre-trained tokenizer to use for processing the inner text.
        code_maps (List[CodeMap)]:
            A list of CodeMap, each specifying Tthe XML-to-code mapping of label codes to specific combinations of tag name and attribute values.
        max_length (int):
            Maximum number of token in one example. Examples will be truncated.
    """
    def __init__(
        self,
        source_dir_path: str,
        dest_dir_path: str,
        code_maps: List[CodeMap],
        tokenizer: AutoTokenizer = config.tokenizer,
        max_length: int = config.max_length,
        subsets: List[str] = ["train", "eval", "test"]
    ):
        self.source_dir_path = Path(source_dir_path)
        self.subsets = subsets
        self.dest_dir_path = dest_dir_path
        if not self.dest_dir_path:
            basename = self.source_dir_path.name
            self.dest_dir_path = Path("/data/json") / basename
        else:
            self.dest_dir_path = Path(self.dest_dir_path)
        if self.dest_dir_path.exists():
            raise ValueError(f"{self.dest_dir_path} already exists! Will not overwrite pre-existing dataset.")
        elif not self.dest_dir_path.parents[0].exists():
            raise ValueError(f"{self.dest_dir_path.parents[0]} does not exist, cannot proceed")
        else:
            self.dest_dir_path.mkdir()
            print(f"{self.dest_dir_path} created")
        self.code_maps = code_maps
        self.max_length = max_length
        self.tokenizer = tokenizer

    def run(self):
        """Runs the coding and labeling of xml examples.
        Saves the resulting text files to the destination directory.
        """
        for subset in self.subsets:
            print(f"Preparing: {subset}")
            source_file_path = self.source_dir_path / f"{subset}.txt"
            dest_file_path = self.dest_dir_path / f"{subset}.jsonl"
            examples = []
            with source_file_path.open() as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    xml_example: Element = fromstring(line)
                    tokens, token_level_labels, text = self._encode_example(xml_example)
                    examples.append({
                        'words': tokens,
                        'text': text,
                        'label_ids': token_level_labels})
            self._save_json(examples, dest_file_path)

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml)
        inner_text = innertext(xml_encoder.element)
        token_level_labels_dict = {}

        for code_map in self.code_maps:
            xml_encoded = xml_encoder.encode(code_map)
            print(xml_encoded)
            stop



            if code_map.name != "panel_start":
                char_level_labels = xml_encoded['label_ids']
                words, token_level_labels = self._from_char_to_token_level_labels(code_map,
                                                                                  list(inner_text),
                                                                                  char_level_labels)
            else:
                char_level_labels = ["O"] * len(xml_encoded['label_ids'])
                offsets = xml_encoded["offsets"]
                for offset in offsets:
                    char_level_labels[offset[0]] = "B-PANEL_START"
                words, token_level_labels = self._from_char_to_token_level_labels_panel(list(inner_text),
                                                                                        char_level_labels)

            token_level_labels_dict[code_map.name] = token_level_labels

        return words,  token_level_labels_dict, inner_text

    def _from_char_to_token_level_labels(self, code_map: CodeMap, text: List[str], labels: List) -> List: # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying Tthe XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            text List[str]:     List of the characters inside the text of the XML elements
            labels List:        List of labels for each character inside the XML elements. They will be
                                a mix of integers and None

        Returns:
            List[str]           Word-level tokenized labels for the input text
        """

        word, label_word = '', ''
        word_level_words, word_level_labels = [], []

        for i, char in enumerate(text):
            if char.isalnum():
                word += char
                label_word += str(labels[i]).replace("None", "O")
            elif char == " ":
                if word not in [""]:
                    word_level_words.append(word)
                    word_level_labels.append(label_word[0])
                word = ''
                label_word = ''
            else:
                if word not in [""]:
                    word_level_words.append(word)
                    word_level_labels.append(label_word[0])

                word_level_words.append(char)
                word_level_labels.append(str(labels[i]).replace("None", "O"))
                word = ''
                label_word = ''

        word_level_iob2_labels = self._labels_to_iob2(code_map, word_level_words, word_level_labels)
        assert len(word_level_words) == len(word_level_iob2_labels), "Length of labels and words not identical!"
        return word_level_words, word_level_iob2_labels

    @staticmethod
    def _from_char_to_token_level_labels_panel(text: List[str], labels: List) -> List: # Checked
        """
        Args:
            text List[str]:     List of the characters inside the text of the XML elements
            labels List:        List of labels for each character inside the XML elements. They will be
                                a mix of integers and None

        Returns:
            List[str]           Word-level tokenized labels for the input text
        """

        word_level_words, word_level_labels = [], []
        word, label_word = '', ''

        for i, char in enumerate(text):
            if char.isalnum():
                word += char
                label_word += str(labels[i])
            elif char == " ":
                if word not in [""]:
                    word_level_words.append(word)
                    if "B-PANEL_START" in label_word:
                        word_level_labels.append("B-PANEL_START")
                    else:
                        word_level_labels.append("O")
                word = ''
                label_word = ''
            else:
                if word not in [""]:
                    word_level_words.append(word)
                    if "B-PANEL_START" in label_word:
                        word_level_labels.append("B-PANEL_START")
                    else:
                        word_level_labels.append("O")
                word_level_words.append(char)
                word_level_labels.append(labels[i])
                word = ''
                label_word = ''

        return word_level_words, word_level_labels


    @staticmethod
    def _labels_to_iob2(code_map: CodeMap, words: List[str], labels: List) -> List: # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying The XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            text List[str]:     List of separated words
            labels List:        List of labels for each word inside the XML elements.

        Returns:
            List[str]           Word-level tokenized labels in IOB2 format

        """
        iob2_labels = []

        for idx, label in enumerate(labels):
            if code_map.name == "panel_start":
                iob2_labels.append("O")

            if code_map.name != "panel_start":
                if label == "O":
                    iob2_labels.append(label)

                if label != "O":
                    if idx == 0:
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2 - 1])
                    if (idx > 0) and (labels[idx - 1] != label):
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2 - 1])
                    if (idx > 0) and (labels[idx - 1] == label):
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2])

        return iob2_labels

    @staticmethod
    def _save_json(examples: List, dest_file_path: Path):
        # saving line by line to json-line file
        with dest_file_path.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
            shuffle(examples)
            for example in examples:
                f.write(f"{json.dumps(example)}\n")

    @staticmethod
    def _cleaning_rules(text, alternative):
        if '"button"' in text:
            text = alternative
        to_replace = {
            "’": "'",
            "–": "-",
            "\xa0": " ",
            "&amp;": "&",
            "&amp;amp;": "&",
            "&nbsp;": " ",
            "\u2009": " ",
            "&gt;": ">",
            "&lt;": "<",
            "amp;": "",
            ") ": ")",
        }

        text = text.strip()
        for x, y in to_replace.items():
            text = text.replace(x, y)

        return text


class PreparatorCharacterTOKCL:
    """Processes source xml documents into examples that can be used in a token classification task.
    It generates the examples on a character-level. This means, that the labels will be generated for each character.
    This can be used with models such as CANINE. No tokenizer is needed in this approach.
    The XML is used to generate labels according to the provided CodeMap.
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_file_path (Path):
            The path to the source file.
        dest_file_path (Path):
            The path of the destination file where the files with the encoded labeled examples should be saved.
        code_maps (List[CodeMap)]:
            A list of CodeMap, each specifying Tthe XML-to-code mapping of label codes to specific combinations of tag name and attribute values.
        max_length (int):
            Maximum number of token in one example. Examples will be truncated.
    """
    def __init__(
        self,
        source_dir_path: str,
        dest_dir_path: str,
        code_maps: List[CodeMap],
        max_length: int = config.max_length,
        subsets: List[str] = ["train", "eval", "test"]
    ):
        self.source_dir_path = Path(source_dir_path)
        self.subsets = subsets
        self.dest_dir_path = dest_dir_path
        if not self.dest_dir_path:
            basename = self.source_dir_path.name
            self.dest_dir_path = Path("/data/json") / basename
        else:
            self.dest_dir_path = Path(self.dest_dir_path)
        if self.dest_dir_path.exists():
            raise ValueError(f"{self.dest_dir_path} already exists! Will not overwrite pre-existing dataset.")
        elif not self.dest_dir_path.parents[0].exists():
            raise ValueError(f"{self.dest_dir_path.parents[0]} does not exist, cannot proceed")
        else:
            self.dest_dir_path.mkdir()
            print(f"{self.dest_dir_path} created")
        self.code_maps = code_maps
        self.max_length = max_length

    def run(self):
        """Runs the coding and labeling of xml examples.
        Saves the resulting text files to the destination directory.
        """
        for subset in self.subsets:
            print(f"Preparing: {subset}")
            source_file_path = self.source_dir_path / f"{subset}.txt"
            dest_file_path = self.dest_dir_path / f"{subset}.jsonl"
            examples = []
            with source_file_path.open() as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    xml_example: Element = fromstring(line)
                    text, char_level_labels = self._encode_example(xml_example)
                    examples.append({
                        'text': text,
                        'label_ids': char_level_labels})
            self._save_json(examples, dest_file_path)

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml)
        inner_text = innertext(xml_encoder.element)
        char_level_labels_dict = {}

        for code_map in self.code_maps:
            entity_labels_element = xml_encoder.encode(code_map)
            if code_map.name != "panel_start":
                char_level_labels = entity_labels_element["label_ids"]
                char_level_labels = ['O' if i is None else i for i in char_level_labels]
                char_level_labels = self._labels_to_iob2(code_map, char_level_labels)
            else:
                char_level_labels = ["O"] * len(entity_labels_element['label_ids'])
                offsets = entity_labels_element["offsets"]
                for offset in offsets:
                    char_level_labels[offset[0]] = "B-PANEL_START"
                    
            char_level_labels_dict[code_map.name] = char_level_labels
                    
        return inner_text,  char_level_labels_dict

    @staticmethod
    def _labels_to_iob2(code_map: CodeMap, labels: List) -> List: # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying The XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            labels List:        List of labels for each word inside the XML elements.

        Returns:
            List[str]           Word-level tokenized labels in IOB2 format

        """
        iob2_labels = []
 
        for idx, label in enumerate(labels):
            if code_map.name == "panel_start":
                iob2_labels.append("O")

            if code_map.name != "panel_start":
                if label == "O":
                    iob2_labels.append(label)

                if label != "O":
                    if idx == 0:
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2 - 1])
                    if (idx > 0) and (labels[idx - 1] != label):
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2 - 1])
                    if (idx > 0) and (labels[idx - 1] == label):
                        iob2_labels.append(code_map.iob2_labels[int(label) * 2])

        return iob2_labels

    @staticmethod
    def _save_json(examples: List, dest_file_path: Path):
        # saving line by line to json-line file
        with dest_file_path.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
            shuffle(examples)
            for example in examples:
                f.write(f"{json.dumps(example)}\n")
