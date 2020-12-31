"""
The CODES dictionary maps a combination of tag name + attribute name + value of the attribute to a unique code.
For example <sd-tag type="cell">...</sd-tag> would receive the code 4, <sd-tag role="normalizing">...</sd-tag> the code 10.
CODES has 4 levels of depth: 
    1. the code family: codes can be used for different situation, marking the whole lengthe of an element or only its boundaries.
    2. the name of the xml tags
    3. the name of the attributes
    4. the value of the attributes.
The codes are provided as values or list of values (in the case that different codes are needed to mark the start and the end of an element, for example).
This is the general structure of CODES:
{
    'code_family': {
        'tag_name': {
            'attribute_name': {
                'attribute_value': code,
                ...
            },
            ...
        },
        ...
    },
    ...
}
"""

CODES = {
    'marks': {
        'sd-tag': {
            'type': {
                '': None, 'molecule': 0, 'gene': 1, 'protein': 2, 'geneprod': 21, 'subcellular': 3, 'cell': 4, 'tissue': 5, 'organism': 6, 'undefined': 7
            },
            'role': {
                '': None, 'intervention': 8, 'assayed': 9, 'normalizing': 10, 'reporter': 11, 'experiment': 12, 'component': 13
            },
            'category': {
                '': None, 'assay': 14, 'entity': 15, 'time': 16, 'physical': 17, 'disease': 18
            }
        }
    },
    'boundaries': {
        'sd-panel': {
            '': {'': [19, 20]}
            }
    }
}
