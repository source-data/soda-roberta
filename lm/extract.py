from typing import List
from pathlib import Path
import argparse
from xml.etree.ElementTree import parse, Element, tostring
from nltk import PunktSentenceTokenizer
from common.utils import cleanup, innertext, progress
from common.config import config
from common import DATASET


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

    def run(self, dest_dir: Path, selector: str, punkt: bool = False, keep_xml: bool = False, remove_tail: bool = True) -> int:
        """Runs the extractor and saves examples in the destination directory.
        A XPath specifies which element to extract from each xml file.
        By default, the inner text from the selected element will be saved as an example.
        It is also possible to prevent this and keep the xml markup, which is useful to train for token classification tasks.
        If sentence tokenization is True, the text is first split into sentences which are individually saved.

        Args:
            dest_dir (Path):
                The path to the desitnation directory.
            selector (str):
                The XPath to select the xml element from which the inner text will be used as example.
            punkt (bool):
                Whether to split the innert text into sentences, which will be saved as individual examples.
            keep_xml (bool):
                Whether to xeep the xml markup instead of extracting innertext. Can be useful for token classification.
            remove_tail (bool):
                Wheter to remove the tail of the xml selected xml element. 

        Returns:
            (int):
                The number of examples saved to disk.
        """

        if not dest_dir.exists():
            dest_dir.mkdir()
            print(f"Created {dest_dir}")
        ext = "xml" if keep_xml else 'txt'
        examples = []
        num_saved_examples = 0
        for i, filepath in enumerate(self.filepaths):
            progress(i, len(self.filepaths), f"{filepath}                         ")
            new_examples = self._examples_from_file(filepath, selector, punkt, keep_xml, remove_tail)
            examples += new_examples
            # save to disk as we go
            for j, example in enumerate(new_examples):
                filename = filepath.stem
                num_saved_examples += self._save(example, dest_dir, filename, str(j), ext)
        print()
        return num_saved_examples

    def _examples_from_file(self, filepath: Path, xpath: str, punkt: bool, keep_xml: bool, remove_tail: bool) -> List[str]:
        examples = []
        elements = self._parse_xml_file(filepath, xpath, remove_tail)
        examples = self._extract_text_from_elements(elements, punkt, keep_xml)
        examples = self._cleanup(examples)
        return examples

    def _parse_xml_file(self, filepath: Path, xpath: str, remove_tail: bool) -> List[str]:
        with filepath.open() as f:
            xml = parse(f)
            elements = xml.findall(xpath)
            if remove_tail:
                for e in elements:
                    if e.tail is not None:
                        print(f"tail: {e.tail} in {e.tag} with text='{e.text}'")
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
        '<xml><b>This was it. Maybe it is not</b></xml>',
        '<xml><b>This <g>is</g> not.</b> It!</xml>'
    ]
    expected_examples = ['This was it.', 'Maybe it is not', 'This is not.']
    for i, e in enumerate(content):
        p = Path('/tmp/test_file_'+str(i)+'.xml')
        p.write_text(e)
    created_filenames = []
    try:
        config.min_char_length = 5
        xtract = ExtractorXML(Path('/tmp'))
        xtract.run(Path('/tmp/test'), selector='.//b', punkt=True)
        created_filenames = [f.name for f in Path('/tmp/test').iterdir()]
        print("created files:", created_filenames)
        expected_filenames = ['test_file_0_0.txt', 'test_file_0_1.txt', 'test_file_1_0.txt']
        assert len(expected_filenames) == len(created_filenames), f"{len(expected_filenames)} <> {len(created_filenames)}"

        for created in created_filenames:
            assert created in expected_filenames, f"'{created}' not it '{expected_filenames}'"
        print("correctly created files!")

        for i, filename in enumerate(created_filenames):
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
    parser.add_argument('destination', nargs="?", default=DATASET, help='Destination folder for extracted text files.')
    parser.add_argument('-S', '--sentences', action='store_true', help='Use this flag to extract individual sentence form each xml element specified by --XPAth.')
    parser.add_argument('-P', '--xpath', default='.//abstract', help='XPath to element to be extracted from XML file.')

    args = parser.parse_args()
    extract_sentences = args.sentences
    xpath = args.xpath
    if not args.corpus:
        self_test()
    else:
        source_path = Path(args.corpus)
        destination_path = Path(args.destination)
        N = ExtractorXML(source_path).run(destination_path, xpath, punkt=extract_sentences)
        print(f"Saved {N} examples.")


if __name__ == '__main__':
    main()
