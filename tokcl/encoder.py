from xml.etree.ElementTree import Element, fromstring
from typing import List
from .xmlcode import CodeMap, EntityTypeCodeMap
from common.utils import innertext


class XMLEncoder:
    """Encodes a XML object in a list of label ids based on the XML-to-label mapping provided by the supplied CodeMap.

    Args:
        code_map (CodeMap):
            The CodeMap object that maps label codes (int) to specic combinations of tag name and attribute values.
    """
    def __init__(self, code_map: CodeMap):
        self.code_map = code_map

    def encode(self, element: Element):
        """Encodes an Element into a list of character-level label codes (int).
        Positions that are not assigned with any code are filled with None.
        THe XML tree is traversed recursively and as soon as an element satistifes to one constraints provided in code_map, 
        the entire span of the element is assigned this code.
        To visualize run:
            python -m tokcl.encoder
        without any arguments.

        Args:
            element (Element):
                The XML Element to encode

        Returns:
            (Dict[List[int], List[Tubple[int, int]]]):
                A dictionary with the list of label ids and the offsets indicating the start and end postition of each labeled element
        """
        offsets = []
        encoded, _ = self._encode(element, offsets)
        labels_and_offsets = {'label_ids': encoded, 'offsets': offsets}
        return labels_and_offsets

    def _encode(self, element: Element, offsets: List = [], pos: int = 0) -> List[int]:
        text_element = element.text or ''
        L_text = len(text_element)
        text_tail = element.tail or ''
        L_tail = len(text_tail)
        code = self._get_code(element)
        L_tot = len(innertext(element))
        if code:
            # as soon as an element corresponds to one of the code, the code is proagated on the whole length of the element and its tail
            encoded = [code] * L_tot
            offsets.append((pos, pos + L_tot))
        else:
            encoded = [None] * L_text
            pos += L_text
            # check child elements
            for child in list(element):
                child_encoded, pos = self._encode(child, offsets=offsets, pos=pos)
                encoded += child_encoded
        encoded = encoded + [None] * L_tail
        pos += L_tot + L_tail
        return encoded, pos

    def _get_code(self, element: Element) -> int:
        for code, constraint in self.code_map.constraints.items():
            if element.tag == constraint['tag']:
                if constraint['attributes']:
                    attributes_found = set(constraint['attributes'].keys()) & set(element.attrib)
                    for a in attributes_found:
                        val = element.attrib[a]
                        if val and val in constraint['attributes'][a]:
                            return code
                else:  # no constraints beyond the tag name
                    return code
        # the element does not match any of the constraints
        return None


def demo():
    example = "<xml>Here <sd-panel>it is: <i>nested in <sd-tag category='entity' type='gene' role='assayed'>Creb-1</sd-tag> with some <sd-tag type='cell'>tail</sd-tag></i>. End</sd-panel>.</xml>"
    xml = fromstring(example)
    xe = XMLEncoder(EntityTypeCodeMap)
    encoded = xe.encode(xml)
    inner_text = innertext(xml)
    assert len(encoded['label_ids']) == len(inner_text)
    text = ''.join([c + '  ' for c in inner_text])
    print("\nExample xml:\n")
    print(example)
    print("\nInner text and features with codes:\n")
    print(text)
    trace = []
    trace = [f"{c:02}" if c is not None else '__' for c in encoded['label_ids']]
    print(f"{' '.join(trace)}")
    print(f"\nOffsets of the labeled elements with their code:")
    for start, end in encoded['offsets']:
        print(f"'{inner_text[start:end]}': start={start}, end={end}, with codes: {encoded['label_ids'][start:end]}")


if __name__ == '__main__':
    demo()
