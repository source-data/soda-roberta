import argparse
from ...xml2labels import SourceDataCodes as sd
from ...dataprep import GeneralTOKCL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the conversion of xml documents into datasets ready for NER learning tasks. It deos it splitting the text into words, without using a pretrained tokenizer. This ensures a more general use of HuggingFace.")
    parser.add_argument("source_dir", help="Directory where the xml files are located.")
    parser.add_argument("dest_dir", help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.BORING, sd.PANELIZATION, sd.SMALL_MOL_ROLES]
    prep = GeneralTOKCL(
        source_dir,
        dest_dir,
        code_maps
    )
    prep.run()
