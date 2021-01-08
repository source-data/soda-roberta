from xml.etree.ElementTree import Element, fromstring, tostring
from typing import List
from .xmlcode import CodeMap, SourceDataCodes as sd
from common.utils import innertext


class XMLEncoder:
    """Encodes a XML object in a list of label ids based on the XML-to-label mapping provided by the supplied CodeMap.

    Args:
            element (Element):
                The XML Element to encode
    """

    def __init__(self, element: Element):
        self.element = element

    def encode(self, code_map: CodeMap):
        """Encodes an Element into a list of character-level label codes (int).
        Positions that are not assigned with any code are filled with None.
        THe XML tree is traversed recursively and as soon as an element satistifes to one constraints provided in code_map, 
        the entire span of the element is assigned this code.
        To visualize run:
            python -m tokcl.encoder
        without any arguments.

        Args:
            code_map (CodeMap):
                The CodeMap object that maps label codes (int) to specic combinations of tag name and attribute values.

        Returns:
            (Dict[List[int], List[Tuple[int, int]]]):
                A dictionary with:
                   - 'label_ids' (List): the list of label ids
                   - 'offsets' (Tuple[int, int]): the offsets indicating the start and end postition of each labeled element
        """
        self.code_map = code_map
        offsets = []
        encoded, _ = self._encode(self.element, offsets)
        labels_and_offsets = {'label_ids': encoded, 'offsets': offsets}  # with runs of spaces reduced to single space
        for start, end in offsets:
            assert encoded[start] == encoded[end-1], f"{encoded[start:end]}\n{start}, {end},\n{innertext(self.element)}\n{innertext(self.element)[start:end]}\n{tostring(self.element)}"
        return labels_and_offsets

    def _encode(self, element: Element, offsets: List = [], pos: int = 0) -> List[int]:
        text_element = element.text or ''
        L_text = len(text_element)
        text_tail = element.tail or ''
        L_tail = len(text_tail)
        code = self._get_code(element)
        inner_text = innertext(element)
        L_inner_text = len(inner_text)
        # print(f"{'•' * pos}{inner_text}<{element.tag}, {pos}, {pos + L_inner_text}, tail={L_tail}>")
        if code:
            # as soon as an element corresponds to one of the code, the code is proagated on the whole length of the element and its tail
            encoded = [code] * L_inner_text
            offsets.append((pos, pos + L_inner_text))
            pos += L_inner_text
        else:
            encoded = [None] * L_text
            pos += L_text
            # check child elements
            for child in element:
                child_encoded, pos = self._encode(child, offsets=offsets, pos=pos)
                encoded += child_encoded
        encoded = encoded + [None] * L_tail
        pos += L_tail
        return encoded, pos

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


def demo():
    example = "<xml><span>Here</span> <sd-panel> it is: <i>nested in <sd-tag category='entity' type='gene' role='reporter'>Creb-1</sd-tag> with some <sd-tag type='protein' role='assayed'>tail</sd-tag></i>. End</sd-panel>.</xml>"
    xml = fromstring(example)
    xe = XMLEncoder(xml)
    inner_text = innertext(xe.element)  # the xml should have no runs of space inside elements.
    encoded = xe.encode(sd.ENTITY_TYPES)
    assert len(encoded['label_ids']) == len(inner_text)
    text4viz = ''.join([c + '  ' for c in inner_text])
    print("\nExample xml:\n")
    print(example)
    print("\nReduced inner text and aligned features with codes:\n")
    print(text4viz)
    trace = []
    trace = [f"{c:02}" if c is not None else '__' for c in encoded['label_ids']]
    print(f"{' '.join(trace)}")
    print(f"\nOffsets of the labeled elements with their code:")
    for start, end in encoded['offsets']:
        print(f"'{inner_text[start:end]}': start={start}, end={end}, with codes: {encoded['label_ids'][start:end]}")


if __name__ == '__main__':
    demo()
