from collections import OrderedDict
from dataclasses import dataclass
from typing import List

"""
CodeMap is used to encode XML elements for token classification tasks. 
CodeMaps map unique codes to a specific set of conditions an XML tag needs to satisfie to be labeled with this code.
For each code, an XML element will be assign this code if:
- it has the specified tag name,
- AND the listed attributes are ALL present,
- AND the attribute have a value IN the provided list.
Each code as a label (str) that can be used when encoding the features of the dataset.

For example, with the constraints held in EntityTypeCodeMap, the element <sd-tag type='protein'>...</sd-tag> will be labeled with code 2.
With PanelBoundaryCodeMap any element <sd-panel>...</sd-panel> will be labeled with code 1, without any furter constaints on attributes and their values.
Usage: call `python -m tokcl.encoder` for a demo.
"""


@dataclass
class CodeMap:
    """Base class. The constraints will be held in constraints.
    """
    constraints: OrderedDict = None

    def __post_init__(self):
        self.all_labels = [c['label'] for c in self.constraints.values()]

    def __len__(self) -> int:
        return len(self.constraints)

"""CodeMap that holds codes to label the role of gnene products, according to the SourceData nomenclature.
"""
GeneprodRoleCodeMap = CodeMap(
    constraints=OrderedDict({
        1: {
            'label': 'controlled var',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['gene', 'protein', 'geneprot'],
                'role': ['intervention'],
            }
        },
        2: {
            'label': 'measured var',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['gene', 'protein', 'geneprot'],
                'role': ['assayed'],
            }
        },
    })
)

"""CodeMap that holds codes to label entities of 8 types, according to the SourceData nomenclature.
"""
EntityTypeCodeMap = CodeMap(
    constraints=OrderedDict({
        1: {
            'label': 'small_molecule',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['small_molecule'],
            }
        },
        2: {
            'label': 'geneprod',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['gene', 'protein', 'geneprod'],
            }
        },
        3: {
            'label': 'subcellular',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['subcellular'],
            }
        },
        4: {
            'label': 'cell',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['cell'],
            }
        },
        5: {
            'label': 'tissue',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['tissue'],
            }
        },
        6: {
            'label': 'organism',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['organism'],
            }
        },
        7: {
            'label': 'exp assay',
            'tag': 'sd-tag',
            'attributes': {
                'type': ['assay'],
            }
        }
    })
)


PanelBoundaryCodeMap = CodeMap(
    constraints=OrderedDict({
        1: {
            'label': 'panel',
            'tag': 'sd-panel',
        },
    })
)
