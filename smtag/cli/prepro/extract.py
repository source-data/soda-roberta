import argparse
import pdb
from ...extract import ExtractorXML


def main():
    parser = argparse.ArgumentParser(description='Extracts datsets from documents.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', default=None, help='path to the corpus of documents to use.')
    parser.add_argument('destination_dir', nargs="?", default=None, help='Destination folder for extracted text files.')
    parser.add_argument('-S', '--sentence_level', action='store_true', help='Use this flag to extract individual sentence form each xml element specified by --XPAth.')
    parser.add_argument('-P', '--xpath', default='.//abstract', nargs="+", help='Space-delimited list of XPath to element to be extracted from XML file.')
    parser.add_argument('-X', '--keep_xml', action="store_true", help='Flag to keep the xml markup.')
    parser.add_argument('-F', '--filter_xpath', default="", type=str, help='Removes XML elements with no elements matching the path defined.')
    parser.add_argument('--inclusion_probability', default=1.0, type=float, help='Probability with which an example will be included.')

    args = parser.parse_args()
    corpus = args.corpus
    destination_dir = args.destination_dir
    sentence_level = args.sentence_level
    xpath = args.xpath
    keep_xml = args.keep_xml
    inclusion_probability = args.inclusion_probability
    if len(xpath) == 1:
        xpath = xpath[0]
    x = ExtractorXML(
        corpus,
        destination_dir=destination_dir,
        sentence_level=sentence_level,
        xpath=xpath,
        keep_xml=keep_xml,
        inclusion_probability=inclusion_probability,
        filter_xpath=(None if args.filter_xpath == "" else args.filter_xpath)
    )
    saved = x.extract_from_corpus()
    print("; \n".join([f"{str(k)}: {v[0]} examples, {v[1]-v[0]} filtered" for k, v in saved.items()]))


if __name__ == '__main__':
    main()
