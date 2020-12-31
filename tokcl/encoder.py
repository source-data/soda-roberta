from xml.etree.ElementTree import Element, fromstring
from typing import Tuple, Dict
from common.utils import innertext
from .xmlcode import CODES


def featurize(element: Element) -> Tuple[Dict, int, int]:
    text_core = element.text or ''
    L_text = len(text_core)
    text_tail = element.tail or ''
    L_tail = len(text_tail)
    features = longitudinal_marks(element, L_text)
    L_tot = L_text
    # add marks recursively
    for child in list(element):
        child_features, L_child_text, L_child_tail = featurize(child)
        L_tot = L_tot + L_child_text + L_child_tail
        # adding to child the features inherited from parent element
        child_features = longitudinal_marks(element, L_child_text, child_features)
        child_tail_features = longitudinal_marks(element, L_child_tail)
        # append the features of child and its tail to the element features generated so far
        for kind in features:
            for e in features[kind]:
                for a in features[kind][e]:
                    features[kind][e][a] += child_features[kind][e][a] + child_tail_features[kind][e][a]
    # add boundaries to current element
    try:
        features = boundary_marks(element, features, L_tot)
    except Exception as e:
        print(element.text, element.tag, element.attrib)
        print(features)
        print(L_tot)
        raise(e)
    return features, L_tot, L_tail


def longitudinal_marks(element: Element, L: int, features: Dict = {}) -> Dict:
    tag = element.tag
    # initialization of the features dict if no features were coded before from a parent element
    if not features:
        features = {
            kind: {
                el: {
                    attr: [None] * L for attr in CODES[kind][el]
                } for el in CODES[kind]
            } for kind in CODES
        }

    if tag in CODES['marks']:
        # marks that do not depend on the presence of any attributes
        if '' in CODES['marks'][tag]:
            features['marks'][tag][''] = [CODES['marks'][tag]['']] * L
        # find the set of attributes of the element that can be encoded
        attributes_found = set(CODES['marks'][tag].keys()) & set(element.attrib)
        for attribute in attributes_found:
            val = element.attrib[attribute]
            if val and val in CODES['marks'][tag][attribute]:
                features['marks'][tag][attribute] = [CODES['marks'][tag][attribute][val]] * L
    return features


def boundary_marks(element, features, L):
    element_tag = element.tag
    if element_tag in CODES['boundaries'] and L > 0:
        if '' in CODES['boundaries'][element_tag]:
            features['boundaries'][element_tag][''][0] = CODES['boundaries'][element_tag][''][''][0]
            features['boundaries'][element_tag][''][L-1] = CODES['boundaries'][element_tag][''][''][1]
        for attribute in (set(CODES['boundaries'][element_tag].keys()) & set(element.attrib)):
            val = element.attrib[attribute]
            if val and val in CODES['boundaries'][element_tag][attribute]:
                features['boundaries'][element_tag][attribute][0] = CODES['boundaries'][element_tag][attribute][val][0]
                features['boundaries'][element_tag][attribute][L-1] = CODES['boundaries'][element_tag][attribute][val][1]
    return features


def demo():
    example = "<xml>Here <sd-panel>it is: <i><sd-tag category='entity' type='gene' role='assayed'>Creb-1</sd-tag></i>. End</sd-panel>.</xml>"
    xml = fromstring(example)
    features, L, _ = featurize(xml)
    inner_text = innertext(xml)
    assert L == len(inner_text)
    text = ''.join([c + '  ' for c in inner_text])
    print("\nExample xml:\n")
    print(example)
    print("\nInner text and features with codes:\n")
    print(text)
    for kind in features:
        for tag in features[kind]:
            for attribute in features[kind][tag]:
                codes = features[kind][tag][attribute]
                trace = []
                for c in codes:
                    if c is None:
                        trace.append('__')
                    else:
                        trace.append(f"{c:02}")
                print(f"{' '.join(trace)}\t{kind} {tag} {attribute}")


if __name__ == '__main__':
    demo()
