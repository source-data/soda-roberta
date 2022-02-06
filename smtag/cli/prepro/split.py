import argparse
from pathlib import Path
from smtag.split import distribute


def main():
    parser = argparse.ArgumentParser(description='Splitting a corpus into train, valid and testsets.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', help='path to the corpus of documents to use.')
    parser.add_argument('-X', '--extension', default='xml', help='Extension (WITHOUT THE DOT) for allowed files in the corpus.')
    args = parser.parse_args()
    corpus = args.corpus
    ext = args.extension
    distribute(Path(corpus), ext=ext)


if __name__ == '__main__':
    main()
