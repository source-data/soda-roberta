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

    def encode(self, element: Element) -> List[int]:
        """Encodes an Element into a list of character-level label codes (int).
        Positions that are not assigned with any code are filled with None.
        THe XML tree is traversed recursively and as soon as an element satistifes to one constraints provided in code_map, 
        the entire span of the element is assigned this code.
        To visualize run:
            python -m tokcl.encoder
        without any arguments.
        """
        text_element = element.text or ''
        L_text = len(text_element)
        text_tail = element.tail or ''
        L_tail = len(text_tail)
        code = self._get_code(element)
        if code:
            # as soon as an element corresponds to one of the code, the code is proagated on the whole length of the element and its tail
            L_tot = len(innertext(element))
            encoded = [code] * L_tot
        else:
            L_tot = L_text
            encoded = [None] * L_text
            # check child elements
            for child in list(element):
                child_encoded = self.encode(child)
                encoded += child_encoded
        encoded = encoded + [None] * L_tail
        return encoded

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
    assert len(encoded) == len(inner_text)
    text = ''.join([c + '  ' for c in inner_text])
    print("\nExample xml:\n")
    print(example)
    print("\nInner text and features with codes:\n")
    print(text)
    trace = []
    trace = [f"{c:02}" if c is not None else '__' for c in encoded]
    print(f"{' '.join(trace)}")


if __name__ == '__main__':
    demo()
