from typing import List
from pathlib import Path
import argparse
from lxml.etree import XPath, tostring, parse, Element
from nltk import PunktSentenceTokenizer
from .utils import cleanup, innertext, progress
from .config import config


class ExtractorXML:
    """Extract multiple text examples from xml documents based on an XPath selector.
    Examples are saved as individual text files.

    Args:
        source_dir (Path):
            the path to the directors that contains the xml files.
    """

    ALLOWED_EXTENSION = ['.xml', '.XML', '.nxml']

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.filepaths = [f for f in self.source_dir.iterdir() if f.suffix in self.ALLOWED_EXTENSION]
        print(f"found {len(self.filepaths)} files.")
        self.xpath = None

    def run(self, dest_dir: Path, selector: XPath, punkt: bool = False, keep_xml: bool = False, remove_tail: bool = True) -> int:
        """Extracts the examples from XML files and saves them in the destination directory.
        The XPath specifies which element to extract from each xml file.
        By default, the inner text from the selected element will be saved as an example.
        It is also possible to prevent this and keep the xml markup, which is useful to train token classification tasks.
        If sentence tokenization is True, the text is first split into sentences which are individually saved.

        Args:
            dest_dir (Path):
                The path to the desitnation directory.
            selector (XPath):
                The XPath to select the xml element from which the inner text will be used as example.
            punkt (bool):
                Whether to split the innert text into sentences, which will be saved as individual examples.
            keep_xml (bool):
                Whether to xeep the xml markup instead of extracting innertext. Can be useful for token classification.
            remove_tail (bool):
                Whether to remove the tail of the xml selected xml element. 

        Returns:
            (int):
                The number of examples saved to disk.
        """

        if not dest_dir.exists():
            dest_dir.mkdir()
            print(f"Created {dest_dir}")
        ext = "xml" if keep_xml else 'txt'
        self.xpath = selector
        num_saved_examples = 0
        for i, filepath in enumerate(self.filepaths):
            progress(i, len(self.filepaths), f"{filepath}                         ")
            new_examples = self._examples_from_file(filepath, punkt, keep_xml, remove_tail)
            # save to disk as we go
            for j, example in enumerate(new_examples):
                filename = filepath.stem
                num_saved_examples += self._save(example, dest_dir, filename, str(j), ext)
        print()
        return num_saved_examples

    def _examples_from_file(self, filepath: Path, punkt: bool, keep_xml: bool, remove_tail: bool) -> List[str]:
        examples = []
        elements = self._parse_xml_file(filepath, remove_tail)
        examples = self._extract_text_from_elements(elements, punkt, keep_xml)
        examples = self._cleanup(examples)
        return examples

    def _parse_xml_file(self, filepath: Path, remove_tail: bool) -> List[str]:
        with filepath.open() as f:
            xml = parse(f)
            elements = self.xpath(xml)
            if remove_tail:
                for e in elements:
                    if e.tail is not None:
                        e.tail = None
        return elements

    def _extract_text_from_elements(self, elements: Element, punkt: bool, keep_xml: bool) -> List[str]:
        examples = []
        if keep_xml:
            for e in elements:
                xml_str = tostring(e).decode('utf-8')  # tostring returns bytes
                length = len(innertext(e))
                if length > config.min_char_length:
                    examples.append(xml_str)
        else:
            for e in elements:
                text = innertext(e)
                if punkt:
                    sentences = PunktSentenceTokenizer().tokenize(text=text)
                    filtered_sentences = [s for s in sentences if self._filter(s)]
                    examples += filtered_sentences
                else:
                    if self._filter(text):
                        examples.append(text)
        return examples

    def _cleanup(self, examples: List[str]) -> List[str]:
        examples = [cleanup(e) for e in examples]
        return examples

    def _filter(self, example: str) -> str:
        example = example if len(example) > config.min_char_length else ''
        return example

    def _save(self, text: str, dest_dir: Path, basename: str, suffix: str, ext: str):
        ex_filename = f"{basename}_{suffix}.{ext}"
        saving_path = dest_dir / ex_filename
        if saving_path.exists():
            print(f"{saving_path} already exists. Not overwritten.                                                     ", end="\r", flush=True)
            return 0
        else:
            saving_path.write_text(text)
            return 1


def self_test():
    """Just call the module to sefl-test it.
    """
    content = [
        '<xml><b>This was it. Maybe it is not.</b></xml>',
        '<xml><b>This <g>is</g> not.</b> It!</xml>'
    ]
    expected_examples = ['This was it.', 'Maybe it is not.', 'This is not.']
    for i, e in enumerate(content):
        p = Path('/tmp/test_file_'+str(i)+'.xml')
        p.write_text(e)
    created_filenames = []
    try:
        config.min_char_length = 5
        xtract = ExtractorXML(Path('/tmp'))
        xtract.run(Path('/tmp/test'), selector=XPath('.//b'), punkt=True)
        created_filenames = [f.name for f in Path('/tmp/test').iterdir()]
        print("created files:", created_filenames)
        expected_filenames = ['test_file_0_0.txt', 'test_file_0_1.txt', 'test_file_1_0.txt']
        assert len(expected_filenames) == len(created_filenames), f"{len(expected_filenames)} <> {len(created_filenames)}"

        for created in created_filenames:
            assert created in expected_filenames, f"'{created}' not it '{expected_filenames}'"
        print("correctly created files!")

        for i, filename in enumerate(expected_filenames):
            expected = expected_examples[i]
            p = Path('/tmp/test') / filename
            loaded = p.read_text()
            print(loaded)
            assert expected == loaded, f"'{expected}' <> '{loaded}'"  # hard to do with smartag
        print("This seems to work!")
    finally:
        for i, e in enumerate(content):
            Path('/tmp/test_file_'+str(i)+'.xml').unlink()
        for f in created_filenames:
            filepath = Path('/tmp/test') / f
            filepath.unlink()
        Path('/tmp/test').rmdir()
        print("Cleaned up and removed test/.")


def main():
    parser = argparse.ArgumentParser(description='Extracts datsets from documents.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', nargs="?", default=None, help='path to the corpus of documents to use.')
    parser.add_argument('destination', nargs="?", default=None, help='Destination folder for extracted text files.')
    parser.add_argument('-S', '--sentences', action='store_true', help='Use this flag to extract individual sentence form each xml element specified by --XPAth.')
    parser.add_argument('-P', '--xpath', default='.//abstract', help='XPath to element to be extracted from XML file.')
    parser.add_argument('-X', '--keep_xml', action="store_true", help='Flag to keep the xml markup.')

    args = parser.parse_args()
    extract_sentences = args.sentences
    xpath = XPath(args.xpath)
    keep_xml = args.keep_xml
    if not args.corpus:
        self_test()
    else:
        corpus_path = Path(args.corpus)
        if not args.destination:
            basename = corpus_path.name
            if keep_xml:
                destination = Path("/data/xml") / basename
            else:
                destination = Path("/data/text") / basename
        subsets = ["train", "eval", "test"]
        source_paths = [corpus_path / subset for subset in subsets]
        destination_paths = [destination / subset for subset in subsets]
        if any([True if p.exists() else False for p in destination_paths]):
            print(f"{destination} is not empty and has already {' or '.join(subsets)} sub-directories. Cannot proceed.")
        else:
            if all([True if p.exists() else False for p in source_paths]):
                if not destination.exists():
                    destination.mkdir()
                    print(f"{destination} created!")
                for source_path, destination_path in zip(source_paths, destination_paths):
                    N = ExtractorXML(source_path).run(destination_path, xpath, punkt=extract_sentences, keep_xml=keep_xml)
                    print(f"Saved {N} examples.")
            else:
                print(f"The source {corpus_path} must include {' & '.join(subsets)} sub-directories. Cannot proceed.")



if __name__ == '__main__':
    main()