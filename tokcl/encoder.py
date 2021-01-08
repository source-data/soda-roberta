from xml.etree.ElementTree import Element, fromstring, tostring
from typing import List
from transformers import RobertaTokenizerFast, BatchEncoding
from .xmlcode import CodeMap, SourceDataCodes as sd
from common.utils import innertext


class XMLEncoder:
    """Encodes a XML object in a list of label ids based on the XML-to-label mapping provided by the supplied CodeMap.
    Both character- and token-level codes are produced.

    Args:
        element (Element):
            The XML Element to encode
        tokenizer (RobertaTokenizerFats): the tokenizer to be usd to recursively tokenized the xml.
    """

    def __init__(self, element: Element, tokenizer: RobertaTokenizerFast):
        self.element = element
        self.tokenizer = tokenizer

    def encode(self, code_map: CodeMap, add_special_tokens: bool = True, max_length: int = None, truncation: bool = False):
        """Encodes an Element into a list of character-level label codes (int) as well as token-level IOB labels.
        Character-level positions that are not assigned with any code are filled with None.
        THe XML tree is traversed and tokenized recursively so that tokenization cannot produce tokens that would span across xml elements.

        To visualize run:
            python -m tokcl.encoder
        without any arguments.

        Args:
            code_map (CodeMap):
                The CodeMap object that maps label codes (int) to specic combinations of tag name and attribute values.
            add_special_tokens (bool): to add begining and end of sentece bos_token_id and eos_token_id
            max_length: the maximum length in terms of token. There is no effect if truncation is not set to True. 
            truncation (bool): whether to truncate tokenized element

        Returns:
            (Dict[List[int], List[Tuple[int, int]]]):
                A dictionary with:
                   - 'label_ids' (List): the list of label ids
                   - 'token_input_ids' (List[int]): the list of token ids after recursive tokenization.
                   - 'token_labels' (List[str]): the list of IOB labels in register with the list of tokens.
                   - 'xml_str (str): the stringifyed xml for reference and debugging
        """
        self.code_map = code_map
        char_level_codes, token_input_ids, token_labels = self._encode(self.element)
        if max_length and truncation:
            stop = max_length
            if add_special_tokens:
                stop -= 2  # such that len(token_indput_ids[:stop]) + 2 special tokens = max_length
            token_input_ids = token_input_ids[:stop]
            token_labels = token_labels[:stop]
        if add_special_tokens:
            token_input_ids = [self.tokenizer.bos_token_id] + token_input_ids + [self.tokenizer.eos_token_id]
            token_labels = ["O"] + token_labels + ["O"]
        labels_and_offsets = {
            'char_level_codes': char_level_codes,
            'token_input_ids': token_input_ids,
            'token_labels': token_labels,
            'xml_str': tostring(self.element).decode('utf-8')  # tostring() returns bytes
        }
        return labels_and_offsets

    def _encode(self, element: Element, code: int = None, prefix: str = '') -> List[int]:
        text_element = element.text or ''
        L_text = len(text_element)
        text_tail = element.tail or ''
        L_tail = len(text_tail)
        token_ids = self._tokenize(text_element)
        token_ids_tail = self._tokenize(text_tail)
        num_tokens = len(token_ids)
        token_labels = ["O"] * num_tokens
        if not code:
            code = self._get_code(element)
            prefix = "B"  # for B-egining token according to IOB2 scheme
        if code:
            char_level_codes = [code] * L_text
            if self.code_map.mode == 'whole_entity':  # label all the tokens corresponding to the xml element
                for token_idx in range(num_tokens):
                    token_labels[token_idx] = self._int_code_to_iob2_label(prefix, code, self.code_map)
                    prefix = "I"  # for subsequet I-nside tokens
            elif self.code_map.mode == 'boundary_start':  # label only the B-egining
                if token_labels:
                    token_labels[0] = self._int_code_to_iob2_label(prefix, code, self.code_map)
                    prefix = ""
                    code = None  # to prevent inheriting code in children elements
        else:
            char_level_codes = [None] * L_text
        # check child elements
        for child in element:
            child_char_level_codes, child_token_ids, child_token_labels = self._encode(child, code=code, prefix=prefix)
            char_level_codes += child_char_level_codes
            token_ids += child_token_ids
            token_labels += child_token_labels
        token_ids = token_ids + token_ids_tail
        char_level_codes = char_level_codes + [None] * L_tail
        token_labels = token_labels + ["O"] * len(token_ids_tail)
        return char_level_codes, token_ids, token_labels

    def _get_code(self, element: Element) -> int:
        for code, constraint in self.code_map.constraints.items():
            if element.tag == constraint['tag']:
                if constraint.get('attributes', None) is not None:
                    if all([
                        element.attrib.get(a, None) in allowed_values
                        for a, allowed_values in constraint['attributes'].items()
                    ]):
                        return code
                else:  # no constraints beyond the tag name
                    return code
        # the element does not match any of the constraints
        return None

    @staticmethod
    def _int_code_to_iob2_label(prefix: str, code: int, code_map: CodeMap) -> str:
        label = code_map.constraints[code]['label']
        iob2_label = f"{prefix}-{label}"
        return iob2_label

    def _tokenize(self, text: str) -> BatchEncoding:
        return self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=False
        ).input_ids


def demo():
    # example = "<xml><span>Here</span> <sd-panel> it is: <i>nested in <sd-tag category='entity' type='gene' role='reporter'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End</sd-panel>.</xml>"
    example = "<xml>Here <sd-panel><p>it is<sd-tag role='reporter'> </sd-tag>: <i>nested <sd-tag role='reporter'>in</sd-tag> <sd-tag category='entity' type='gene' role='intervention'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End </p></sd-panel>."
    example += ' 1 2 3 4 5' + '</xml>'
    xml = fromstring(example)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    xe = XMLEncoder(xml, tokenizer)
    inner_text = innertext(xe.element)
    encoded = xe.encode(sd.ENTITY_TYPES, max_length=22, truncation=True, add_special_tokens=True)
    assert len(encoded['char_level_codes']) == len(inner_text)
    text4viz = ''.join([c + '  ' for c in inner_text])
    print("\nExample xml:\n")
    print(example)
    print("\nReduced inner text and aligned features with codes:\n")
    print(text4viz)
    trace = []
    trace = [f"{c:02}" if c is not None else '__' for c in encoded['char_level_codes']]
    print(f"{' '.join(trace)}")
    for idx in range(len(encoded['token_input_ids'])):
        token_id = encoded['token_input_ids'][idx]
        token = xe.tokenizer.convert_ids_to_tokens(token_id)
        token_label = encoded['token_labels'][idx]
        print(f"{idx}:\t{token_id}\t{token}\t{token_label}")


if __name__ == '__main__':
    demo()
