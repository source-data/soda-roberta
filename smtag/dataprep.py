from pathlib import Path
from typing import List, Tuple, Dict
import json
from lxml.etree import fromstring, Element
from random import shuffle
import celery
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding
from .xml2labels import CodeMap
from .encoder import XMLEncoder
from .celery import app
from .config import config
from .utils import innertext


# Celery tasks
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
        truncation=config.truncation,
        return_offsets_mapping=True,
        return_special_tokens_mask=True
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
            with source_file_path.open() as f:
                task_list = []
                lines = f.readlines()
                i = 0
                for line in tqdm(lines):
                    line = line.strip()
                    if line:
                        task_list.append(aligned_tokenization_task.s(line, str(dest_file_path), self.max_length))
                    if i % batch_size == 0:
                        job = celery.group(task_list)
                        results = job.apply_async()
                        results.get()
                        task_list = []
                    i += 1
                job = celery.group(task_list)
                results = job.apply_async()
                results.get()
            self._verify(dest_file_path)

    def _verify(self, dest_file_path: Path):
        with dest_file_path.open() as f:
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
        n += 1
        print("\nLength verification: OK!")
        print(f"\naverage input_ids length = {round(cumul_len / n)} (min={min_len}, max={max_len}) tokens")
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
        split_ratio (Dict[str, float]):
            Proportion of examples in train, eval and test subsets.
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
            labeled_examples = []
            with source_file_path.open() as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    xml_example: Element = fromstring(line)
                    tokenized, token_level_labels = self._encode_example(xml_example)
                    labeled_examples.append({
                        'tokenized': tokenized,
                        'label_ids': token_level_labels
                    })
            self._save_json(labeled_examples, dest_file_path)
            self._verify(dest_file_path)

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml)
        inner_text = innertext(xml_encoder.element)
        tokenized: BatchEncoding = self.tokenizer(
            inner_text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        token_level_labels = {}
        for code_map in self.code_maps:
            xml_encoded = xml_encoder.encode(code_map)
            labels = self._align_labels(tokenized, xml_encoded, code_map, inner_text)
            token_level_labels[code_map.name] = labels
        return tokenized, token_level_labels

    def _align_labels(self, tokenized: BatchEncoding, xml_encoded: Dict, code_map: CodeMap, inner_text) -> List[int]:
        num_tokens = len(tokenized.input_ids)
        # prefill with outside of entity label 'O' using IOB2 scheme
        token_level_labels = ['O'] * num_tokens
        # tokenizer may have truncated the example
        last_token_start, last_token_end = tokenized.offset_mapping[-2]  # -2 because the last token is </s> with offsets (0,0) by convention
        # get the character-level start end end of the xml element and try to map to tokens
        inner_text = inner_text[:last_token_end]
        for element_start, element_end in xml_encoded['offsets']:
            # check we are still within the truncated example
            if (element_start <= last_token_start) & (element_end < last_token_end):
                code = xml_encoded['label_ids'][element_start]  # element_end would give the same, maybe check with assert
                assert xml_encoded['label_ids'][element_start] == xml_encoded['label_ids'][element_end - 1], f"{xml_encoded['label_ids'][element_start:element_end]}\n{element_start, element_end}"
                start_token_idx = self._char_to_token(element_start, inner_text, tokenized)
                end_token_idx = self._char_to_token(element_end, inner_text, tokenized)
                # sanity check
                try:
                    assert start_token_idx is not None, f"\n\nproblem with start token None."
                    assert end_token_idx is not None, f"\n\nproblem with end token None."
                except Exception:
                    import pdb; pdb.set_trace()
                if (start_token_idx == end_token_idx):
                    # In addition, the tokenizer may generate a token that is actually spanning an element boundary.
                    # Recursive tokenized in the XMLEncode is NOT a solution as it will force learning on atypical tokenization that will not be
                    # representative of tokenization of free text. It actually destroys prediction :-(
                    # But empty element cannot not correspond to any token
                    start, end = tokenized.offset_mapping[end_token_idx]
                    if (start <= element_start) and (end > element_end):
                        print(f"WARNING: token overlaps element boundary {code_map.constraints[code]['tag']} at position {element_end} in '{inner_text[start-10:start]}>>>{inner_text[start:element_end]}^{inner_text[element_end:end]}...<<<{inner_text[end:end+10]}'")
                        # if next token outside of an element, will be labeled; if part of next element, labelig will be overriden
                        end_token_idx += 1 if end_token_idx <= num_tokens else num_tokens
                    else:
                        print(f"WARNING: emtpy element {code_map.constraints[code]['tag']}? at position {element_start, element_end} in >>>{inner_text[element_start:element_start+50]}...<<<")
                prefix = "B"  # for B-egining token according to IOB2 scheme
                if code_map.mode == 'whole_entity':  # label all the tokens corresponding to the xml element
                    for token_idx in range(start_token_idx, end_token_idx):
                        label = self._int_code_to_iob2_label(prefix, code, code_map)
                        token_level_labels[token_idx] = label
                        prefix = "I"  # for subsequet I-nside tokens
                elif code_map.mode == 'boundary_start' and (start_token_idx != end_token_idx):  # label the B-egining of non-empty elements
                    label = self._int_code_to_iob2_label(prefix, code, code_map)
                    token_level_labels[start_token_idx] = label
            else:
                # the last token has been reached, no point scanning further elemnts
                break
        return token_level_labels

    @staticmethod
    def _char_to_token(element_pos, inner_text, tokenized):
        # Nasty: because of RobertaTokenizer's behavior with spaces, 
        # a space before a word is included in token. When this happens across xml element boundary, 
        # the character at the boundary position is a space and is included in the next or previous token outside the element.
        # In addition, BatchEncoding.char_to_token() will return None if the token is a single space
        # proper token will be found only from next or previous character, respectively
        # This gymnastic is to try to circumven this.
        pos = element_pos
        # _, last_pos = tokenized.offset_mapping[-2]  # end of last non special token
        if pos >= len(inner_text):
            token_idx = len(tokenized.input_ids)
            return token_idx
        elif inner_text[pos] != ' ':  # usual case, not in a space, all fine
            token_idx = tokenized.char_to_token(pos)
            return token_idx
        while (inner_text[pos] == ' ') and (pos < len(inner_text) - 1): pos += 1  # scanning for non space on the right
        if inner_text[pos] == ' ':  # we are still in a run of space and at the end of the string!
            token_idx = len(tokenized.input_ids) - 1
        else:
            # __.token    is tokenized into two single spaces plus one .token (dot is spacial character produced by RobertaTokenizer)
            #    ^        need to scan until non space character
            # 5           element_start = 5
            # 5678        pos = 8 after scanning
            # 234         actual start_token_idx 2, first non space token is 4, tokens 2 and 3 are single spaces
            num_single_space_tokens = pos - 1 - element_pos
            try:
                token_idx = tokenized.char_to_token(pos) - num_single_space_tokens
            except Exception:
                import pdb; pdb.set_trace()
        return token_idx

    @staticmethod
    def _int_code_to_iob2_label(prefix: str, code: int, code_map: CodeMap) -> str:
        label = code_map.constraints[code]['label']
        iob2_label = f"{prefix}-{label}"
        return iob2_label

    def _save_json(self, examples: List, dest_file_path: Path):
        # saving line by line to json-line file
        with dest_file_path.open('a', encoding='utf-8') as f:  # mode 'a' to append lines
            shuffle(examples)
            for example in examples:
                j = {
                    'tokens': example['tokenized'].tokens(),
                    'input_ids': example['tokenized'].input_ids,
                    'label_ids':  example['label_ids']
                }
                f.write(f"{json.dumps(j)}\n")

    def _verify(self, dest_file_path):
        with dest_file_path.open() as f:
            cumul_len = 0
            max_len = 0
            longest_example = ''
            min_len = 1E3
            shortest_example = ''
            for n, line in enumerate(f):
                j = json.loads(line)
                L = len(j['tokens'])
                assert L <= self.max_length, f"Length verification: error line {n} in {p} with {len(j['tokens'])} tokens > {self.max_length}."
                assert len(j['input_ids']) == L, f"mismatch in number of tokens and input_ids: error line {n} in {p}"
                for k, label_ids in j['label_ids'].items():
                    assert len(label_ids) == L, f"mismatch in number of tokens and {k} label_ids: error line {n} in {p}"
                cumul_len += L
                if L > max_len:
                    max_len = L
                    longest_example = j['input_ids']
                if L < min_len:
                    min_len = L
                    shortest_example = j['input_ids']
        n += 1
        print("\nLength verification: OK!")
        print(f"\naverage input_ids length = {round(cumul_len / n)} (min={min_len}, max={max_len}) tokens")
        print(f"longest example: {config.tokenizer.decode(longest_example)}")
        print(f"shortest example: {config.tokenizer.decode(shortest_example)}")
        return True
