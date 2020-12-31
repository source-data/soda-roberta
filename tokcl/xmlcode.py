"""
CodeMap is used to encode XML elements for token classification tasks. 
CodeMaps map unique codes to a specific set of conditions an XML tag needs to satisfie to be labeled with this code.
For each code, an XML element will be assign this code if:
- it has the specified tag name,
- AND the listed attributes are ALL present,
- AND the attribute have a value IN the provided list.

For example, with the constraints held in EntityTypeCodeMap, the element <sd-tag type='protein'>...</sd-tag> will be labeled with code 2.
With PanelBoundaryCodeMap any element <sd-panel>...</sd-panel> will be labeled with code 1, without any furter constaints on attributes and their values.
Usage: call `python -m tokcl.encoder` for a demo.
"""

class CodeMap:
    constraints = {}


class GeneprodRoleCodeMap(CodeMap):
    constraints = {
        1: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('gene', 'protein', 'geneprot'),
                'role': ('intervention', 'assayed'),
            }
        },
    }


class EntityTypeCodeMap(CodeMap):
    constraints = {
        1: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('small_molecule'),
            }
        },
        2: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('gene', 'protein', 'geneprod'),
            }
        },
        3: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('small_molecule'),
            }
        },

        4: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('subcellular'),
            }
        },
        5: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('cell'),
            }
        },
        6: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('tissue'),
            }
        },
        7: {
            'tag': 'sd-tag',
            'attributes': {
                'type': ('organism'),
            }
        },
    }


class PanelBoundaryCodeMap(CodeMap):

    constraints = {
        1: {
            'tag': 'sd-panel',
        },
    }
