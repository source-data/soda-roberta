from argparse import ArgumentParser
from ...xml2labels import SourceDataCodes as sd
from ...dataprep import PreparatorTOKCL

if __name__ == "__main__":
    parser = ArgumentParser(description="Prepares the conversion of xml documents into datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", help="Directory where the xml files are located.")
    parser.add_argument("dest_dir", help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    code_maps = [sd.ENTITY_TYPES, sd.GENEPROD_ROLES, sd.BORING, sd.PANELIZATION]
    prep = PreparatorTOKCL(
        source_dir,
        dest_dir,
        code_maps
    )
    prep.run()
