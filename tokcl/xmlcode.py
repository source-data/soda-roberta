from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


@dataclass
class CodeMap:
    """CodeMaps are used to encode XML elements for token classification tasks. 
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
    name: str = ''
    constraints: OrderedDict = None

    def __post_init__(self):
        self.all_labels: List[str] = [c['label'] for c in self.constraints.values()]
        self.iob2_labels: List[str] = ['O']  # generate labels of IOB2 schema tagging, including prefix combinations
        for label in self.all_labels:
            for prefix in ['I', 'B']:
                self.iob2_labels.append(f"{prefix}-{label}")

    def from_label(self, label: str) -> Dict:
        idx = self.all_labels.index(label)
        constraint = self.constraints[idx + 1]  # constraints keys start at 1
        return constraint


class SourceDataCodes(Enum):
    """A series of CodeMaps to encode SourceData labeling scheme.

    Properties:
        GENEPROD_ROLE (CodeMap):
            CodeMap that holds codes to label the role of gene products, according to the SourceData nomenclature.
        ENTITY_TYPES (CodeMap):
            CodeMap that holds codes to label entities of 8 types, according to the SourceData nomenclature.
        PANEL_START (CodeMap):
            Start of panel legends within a figure legend.
    """

    @property
    def name(self) -> str:
        """The name of the code map. Will be used as column header or field name in dataset with multiple
        tags.
        """
        return self.value.name

    @property
    def iob2_labels(self) -> List[str]:
        """Returns (List[str]): all the generated IOB2 labels including outside "O".
        """
        return self.value.iob2_labels

    @property
    def all_labels(self) -> List[str]:
        """Returns (List[str]): all labels from the CodeMap.
        """
        return self.value.all_labels

    @property
    def constraints(self) -> OrderedDict:
        """Returns (OrderedDict) all the constraints for each label.
        """
        return self.value.constraints

    def from_label(self, label) -> Dict:
        """Returns (Dict): the constraint corresponding to the given label.
        """
        return self.value.from_label(label)

    GENEPROD_ROLES = CodeMap(
        name="geneprod_roles",
        constraints=OrderedDict({
            1: {
                'label': 'CONTROLLED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['geneprod', 'gene', 'protein'],
                    'role': ['intervention'],
                }
            },
            2: {
                'label': 'MEASURED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['geneprod', 'gene', 'protein'],
                    'role': ['assayed'],
                }
            },
        })
    )

    ENTITY_TYPES = CodeMap(
        name="entity_types",
        constraints=OrderedDict({
            1: {
                'label': 'SMALL_MOLECULE',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['molecule'],
                }
            },
            2: {
                'label': 'GENEPROD',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['geneprod', 'gene', 'protein'],
                }
            },
            3: {
                'label': 'SUBCELLULAR',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['subcellular'],
                }
            },
            4: {
                'label': 'CELL',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['cell'],
                }
            },
            5: {
                'label': 'TISSUE',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['tissue'],
                }
            },
            6: {
                'label': 'ORGANISM',
                'tag': 'sd-tag',
                'attributes': {
                    'type': ['organism'],
                }
            },
            7: {
                'label': 'EXP_ASSAY',
                'tag': 'sd-tag',
                'attributes': {
                    'category': ['assay'],
                }
            }
        })
    )

    PANEL = CodeMap(
        name="panel",
        constraints=OrderedDict({
            1: {
                'label': 'PANEL',
                'tag': 'sd-panel',
            },
        })
    )
