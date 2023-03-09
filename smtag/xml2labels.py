from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


@dataclass
class CodeMap:
    """CodeMaps are used to encode XML elements for token classification tasks. 
    CodeMaps map unique codes to a specific set of conditions an XML tag needs to satisfie to be labeled with this code.
    For each code, an XML element will be assigned this code if:
    - it has the specified tag name,
    - AND the listed attributes are ALL present,
    - AND the attribute have a value IN the provided list.
    Each code as a label (str) that can be used when encoding the features of the dataset.

    For example, with the constraints held in ENTITY_TYPES, the element <sd-tag entity_type='protein'>...</sd-tag> will be labeled with code 2.
    With PanelBoundaryCodeMap any element <sd-panel>...</sd-panel> will be labeled with code 1, without any furter constraints on attributes and their values.
    Usage: call `python -m tokcl.encoder` for a demo.

    Properties:
        name (str): the name of the CodeMap. Useful when serializing codes from several CodeMaps.
        contraints (Dict): the constraintts on tag name and attribute values to assign a code to an xml element.
    """
    name: str = ''
    mode: str = ''
    constraints: OrderedDict = None

    def __post_init__(self):
        self.all_labels: List[str] = [c['label'] for c in self.constraints.values()]
        self.iob2_labels: List[str] = ['O']  # generated labels of IOB2 schema tagging, including prefix combinations
        for label in self.all_labels:
            if self.mode == 'whole_entity':
                for prefix in ['B', 'I']:
                    self.iob2_labels.append(f"{prefix}-{label}")
            elif self.mode == 'boundary_start':
                # only B_egning of entities, no I_nside tag
                self.iob2_labels.append(f"B-{label}")
            else:
                raise ValueError(f"CodeMap mode {self.mode} unkown.")

    def from_label(self, label: str) -> Dict:
        """Returns (Dict): the constraint corresponding to the given label WITHOUT prefix (for example 'GENEPROD' OR 'CONTROLLED_VAR').
        """
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
        BORING (CodeMap):
            CodeMap specifying the attributes of potentially uninteresting entities ('boring').
        PANELIZATION (CodeMap):
            Start of panel legends within a figure legend.
    """

    @property
    def name(self) -> str:
        """The name of the code map. Will be used as column header or field name in dataset with multiple
        tags.
        """
        return self.value.name

    @property
    def mode(self) -> str:
        """Specifies the tagging mode:
            - 'whole_entity': the whole entity will be tagged from the begining (B-prefixed tag) to the end (I-prefixed tags).
            - 'boundary_start': the feature indicate the boudary between text segments and only the begining of the segment will be labeled (with B-prefixed tag)
        """
        return self.value.mode

    @property
    def iob2_labels(self) -> List[str]:
        """Returns (List[str]): all the generated IOB2 labels (WITH prefixes) including outside "O".
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
        """Returns (Dict): the constraint corresponding to the given label WITHOUT prefix (for example 'GENEPROD' OR 'CONTROLLED_VAR').
        """
        return self.value.from_label(label)

    GENEPROD_ROLES = CodeMap(
        name="geneprod_roles",
        mode="whole_entity",
        constraints=OrderedDict({
            1: {
                'label': 'CONTROLLED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['geneprod', 'gene', 'protein'],
                    'role': ['intervention'],
                }
            },
            2: {
                'label': 'MEASURED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['geneprod', 'gene', 'protein'],
                    'role': ['assayed'],
                }
            }
        })
    )

    SMALL_MOL_ROLES = CodeMap(
        name="small_mol_roles",
        mode="whole_entity",
        constraints=OrderedDict({
            1: {
                'label': 'CONTROLLED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['molecule'],
                    'role': ['intervention'],
                }
            },
            2: {
                'label': 'MEASURED_VAR',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['molecule'],
                    'role': ['assayed'],
                }
            }
        })
    )

    ENTITY_TYPES = CodeMap(
        name="entity_types",
        mode="whole_entity",
        constraints=OrderedDict({
            1: {
                'label': 'SMALL_MOLECULE',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['molecule'],
                }
            },
            2: {
                'label': 'GENEPROD',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['geneprod', 'gene', 'protein'],
                }
            },
            3: {
                'label': 'SUBCELLULAR',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['subcellular'],
                }
            },
            # 4: {
            #     'label': 'CELL',
            #     'tag': 'sd-tag',
            #     'attributes': {
            #         'entity_type': ['cell'],
            #     }
            # },
            4: {
                'label': 'CELL_TYPE',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['cell_type'],
                }
            },
            5: {
                'label': 'TISSUE',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['tissue'],
                }
            },
            6: {
                'label': 'ORGANISM',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['organism'],
                }
            },
            7: {
                'label': 'EXP_ASSAY',
                'tag': 'sd-tag',
                'attributes': {
                    'category': ['assay'],
                }
            },
            8: {
                'label': 'DISEASE',
                'tag': 'sd-tag',
                'attributes': {
                    'category': ['disease'],
                }
            },
            9: {
                'label': 'CELL_LINE',
                'tag': 'sd-tag',
                'attributes': {
                    'entity_type': ['cell_line'],
                }
            },
        })
    )

    BORING = CodeMap(
        name="boring",
        mode="whole_entity",
        constraints=OrderedDict({
            1: {
                'label': 'BORING',
                'tag': 'sd-tag',
                'attributes': {
                    'role': ['reporter', 'normalizing', 'component']
                }
            },
        })
    )

    PANELIZATION = CodeMap(
        name="panel_start",
        mode="boundary_start",
        constraints=OrderedDict({
            1: {
                'label': 'PANEL_START',
                'tag': 'sd-panel',
            },
        })
    )


def main():
    _NER_LABEL_NAMES = SourceDataCodes.ENTITY_TYPES.iob2_labels
    print(f"_NER_LABEL_NAMES={_NER_LABEL_NAMES}")
    _SEMANTIC_GENEPROD_ROLES_LABEL_NAMES = SourceDataCodes.GENEPROD_ROLES.iob2_labels
    print(f"_SEMANTIC_ROLES_LABEL_NAMES={_SEMANTIC_GENEPROD_ROLES_LABEL_NAMES}")
    _SEMANTIC_SMALL_MOL_ROLES_LABEL_NAMES = SourceDataCodes.GENEPROD_ROLES.iob2_labels
    print(f"_SEMANTIC_ROLES_LABEL_NAMES={_SEMANTIC_SMALL_MOL_ROLES_LABEL_NAMES}")
    _BORING_LABEL_NAMES = SourceDataCodes.BORING.iob2_labels
    print(f"_BORING_LABEL_NAMES={_BORING_LABEL_NAMES}")
    _PANEL_START_NAMES = SourceDataCodes.PANELIZATION.iob2_labels
    print(f"_PANEL_START_NAMES={_PANEL_START_NAMES}")


if __name__ == "__main__":
    main()
