from xml.etree.ElementTree import Element, fromstring
from typing import List
from .xmlcode import CodeMap, EntityTypeCodeMap
from common.utils import innertext


class XMLEncoder:
    def __init__(self, code_map: CodeMap):
        self.code_map = code_map

    def encode(self, element: Element) -> List:
        text_element = element.text or ''
        L_text = len(text_element)
        text_tail = element.tail or ''
        L_tail = len(text_tail)
        code = self.get_code(element)
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

    def get_code(self, element: Element) -> int:
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


    # def boundary_marks(self, element, features, L):
    #     element_tag = element.tag
    #     if element_tag in CODES['boundaries'] and L > 0:
    #         if '' in CODES['boundaries'][element_tag]:
    #             features['boundaries'][element_tag][''][0] = CODES['boundaries'][element_tag][''][''][0]
    #             features['boundaries'][element_tag][''][L-1] = CODES['boundaries'][element_tag][''][''][1]
    #         for attribute in (set(CODES['boundaries'][element_tag].keys()) & set(element.attrib)):
    #             val = element.attrib[attribute]
    #             if val and val in CODES['boundaries'][element_tag][attribute]:
    #                 features['boundaries'][element_tag][attribute][0] = CODES['boundaries'][element_tag][attribute][val][0]
    #                 features['boundaries'][element_tag][attribute][L-1] = CODES['boundaries'][element_tag][attribute][val][1]
    #     return features


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
