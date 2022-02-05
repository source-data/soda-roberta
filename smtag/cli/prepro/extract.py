from argparse import ArgumentParser
from ...extract import ExtractorXML


def main():
    parser = ArgumentParser(description='Extracts datsets from documents.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', default=None, help='path to the corpus of documents to use.')
    parser.add_argument('destination_dir', nargs="?", default=None, help='Destination folder for extracted text files.')
    parser.add_argument('-S', '--sentence_level', action='store_true', help='Use this flag to extract individual sentence form each xml element specified by --XPAth.')
    parser.add_argument('-P', '--xpath', default='.//abstract', help='XPath to element to be extracted from XML file.')
    parser.add_argument('-X', '--keep_xml', action="store_true", help='Flag to keep the xml markup.')
    parser.add_argument('--inclusion_probability', default=1.0, type=float, help='Probability with which an example will be included.')

    args = parser.parse_args()
    corpus = args.corpus
    destination_dir = args.destination_dir
    sentence_level = args.sentences
    xpath = args.xpath
    keep_xml = args.keep_xml
    inclusion_probability = args.proba

    x = ExtractorXML(
        corpus,
        destination_dir=destination_dir,
        sentence_level=sentence_level,
        xpath=xpath,
        keep_xml=keep_xml,
        inclusion_probability=inclusion_probability
    )
    saved = x.extract_from_corpus()
    print(saved)


if __name__ == '__main__':
    main()
