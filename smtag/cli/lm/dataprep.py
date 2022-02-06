import argparse
from ...dataprep import PreparatorLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize text and prepares the datasets ready for NER learning tasks.")
    parser.add_argument("source_dir", help="Directory where the source files are located.")
    parser.add_argument("dest_dir", help="The destination directory where the labeled dataset will be saved.")
    args = parser.parse_args()
    source_dir_path = args.source_dir
    dest_dir_path = args.dest_dir
    sdprep = PreparatorLM(source_dir_path, dest_dir_path)
    sdprep.run()
