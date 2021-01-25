from pathlib import Path
import argparse
from random import random
import celery
from lxml.etree import XPath
from .tasks import examples_from_file_task, save_task
from .utils import progress
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

    def run(self, dest_dir: Path, selector: str, punkt: bool = False, keep_xml: bool = False, remove_tail: bool = True, inclusion_probability: float = 1.0) -> int:
        """Extracts the examples from XML files and saves them in the destination directory.
        The XPath specifies which element to extract from each xml file.
        By default, the inner text from the selected element will be saved as an example.
        It is also possible to prevent this and keep the xml markup, which is useful to train token classification tasks.
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
                Whether to remove the tail of the xml selected xml element. 
            inclusion_probability (float):
                Probability with which each example is included in the dataset. Allows to only take random subsample of very large dataset.

        Returns:
            (int):
                The number of examples saved to disk.
        """

        ext = "xml" if keep_xml else 'txt'
        num_saved_examples = 0
        batch_size = config.celery_batch_size
        N = len(self.filepaths)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            progress(end-1, len(self.filepaths)-1, f"{start}:{end} of {N}")
            task_list = [
                examples_from_file_task.s(str(filepath), selector, punkt, keep_xml, remove_tail)
                for filepath in self.filepaths[start:end]
            ]
            job = celery.group(task_list)
            results = job.apply_async()
            results = results.get()
            # save to disk as we go
            for res in results:
                new_examples = res['examples']
                filepath = res['filepath']
                filename = Path(filepath).stem
                saving_tasks = []
                for j, example in enumerate(new_examples):
                    proba = random()
                    if proba <= inclusion_probability:
                        saving_tasks.append(save_task.s(example, str(dest_dir), filename, str(j), ext))
                job = celery.group(saving_tasks)
                saving_results = job.apply_async()
                saving_results.get()
                num_saved_examples = len(saving_results)
        print()
        return num_saved_examples


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
    parser.add_argument('--proba', default=1.0, type=float, help='Probability with which an example will be included.')

    args = parser.parse_args()
    extract_sentences = args.sentences
    xpath = args.xpath
    keep_xml = args.keep_xml
    destination = args.destination
    inclusion_probability = args.proba
    if not args.corpus:
        self_test()
    else:
        corpus_path = Path(args.corpus)
        if destination:
            destination_dir = Path(destination)
        else:
            basename = corpus_path.name
            if keep_xml:
                destination_dir = Path("/data/xml") / basename
            else:
                destination_dir = Path("/data/text") / basename
        subsets = ["train", "eval", "test"]
        source_paths = [corpus_path / subset for subset in subsets]
        destination_paths = [destination_dir / subset for subset in subsets]
        if any([True if p.exists() else False for p in destination_paths]):
            print(f"{destination_dir} is not empty and has already {' or '.join(subsets)} sub-directories. Cannot proceed.")
        else:
            if all([True if source.exists() else False for source in source_paths]):
                if not destination_dir.exists():
                    destination_dir.mkdir()
                    print(f"{destination_dir} created!")
                for source_path, destination_path in zip(source_paths, destination_paths):
                    if not destination_path.exists():
                        destination_path.mkdir()
                        print(f"Created {destination_path}")
                    N = ExtractorXML(source_path).run(
                        destination_path,
                        xpath,
                        punkt=extract_sentences,
                        keep_xml=keep_xml,
                        inclusion_probability=inclusion_probability
                    )
                    print(f"Saved {N} examples.")
            else:
                print(f"The source {corpus_path} must include {' & '.join(subsets)} sub-directories. Cannot proceed.")


if __name__ == '__main__':
    main()
