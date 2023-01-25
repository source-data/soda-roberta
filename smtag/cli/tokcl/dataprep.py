import argparse
from ...xml2labels import SourceDataCodes as sd
from ...dataprep import PreparatorTOKCL, PreparatorCharacterTOKCL
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the conversion of xml documents into datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", help="Directory where the xml files are located.")
    parser.add_argument("dest_dir", help="The destination directory where the labeled dataset will be saved.")
    parser.add_argument("-C", "--character_level", action="store_true", help="Generate the data with labels on a character level.")
    parser.add_argument("-P", "--panelization", action="store_true", help="Generate the data with labels on a character level.")
    parser.add_argument("-T", "--tokenizer", default="", help="Tokenizer to be used. Relevant only for roberta.")
    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    character_level = args.character_level
    panelization = args.panelization
    code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.SMALL_MOL_ROLES, sd.PANELIZATION]
    if args.tokenizer != "":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True)
    if character_level:
        if panelization:
            prep = PreparatorCharacterTOKCL(
                source_dir,
                dest_dir,
                [sd.PANELIZATION]
            )
        else:
            prep = PreparatorCharacterTOKCL(
                source_dir,
                dest_dir,
                [sd.ENTITY_TYPES]
            )

    else:
        prep = PreparatorTOKCL(
            source_dir,
            dest_dir,
            code_maps,
            tokenizer=(tokenizer if args.tokenizer != "" else args.tokenizer )
        )
    prep.run()
