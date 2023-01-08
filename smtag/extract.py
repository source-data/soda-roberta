from pathlib import Path
import pdb
from typing import List, Dict, Tuple, Union
from random import random
import celery
from tqdm import tqdm
from lxml.etree import Element, tostring, parse, Error
from nltk import PunktSentenceTokenizer
from .celery import app
from .utils import innertext, cleanup
from .config import config


# Celery tasks
def examples_from_file_task(
    filepath: str,
    dest_file_path: str,
    xpath: Union[str, List[str]],
    punkt: bool,
    keep_xml: bool,
    remove_tail: bool,
    min_length: int = config.min_char_length
) -> int:
    """Generates text or xml examples from xml documents. 
    Examples to be extracted are found using an XPath expression.
    THe resulting text can be segmented in sentences if desired. 
    Either the inner text is extracted or the xml of the extracted element is kept.

    Args:
        indx (int): the index position of the file, useful to keep track of order when asynchonous processing is used
        filepath (str): the path to the source file.
        xpath (Union[str, List[str]]): the XPath expression to identify the example(s) in the xml file.
        punkt (bool): whether to split the text into individual sentences.
        keep_xml (bool): whether to keep the xml of the element, otherwise the inner text is extracted.
        remove_tail (bool): set this to False if the text after the element should be included.
    """
    examples = []
    if isinstance(xpath, str):
        elements = _parse_xml_file(filepath, xpath, remove_tail)
        text = _extract_text_from_elements(elements, punkt, keep_xml, min_length=min_length)
        examples = _cleanup(text)
    elif isinstance(xpath, list):  # twin examples extracted
        twin_examples = []
        if len(xpath) != 2:
            raise ValueError(f"xpath pair has to have exactly 2 elements (len is {len(xpath)}")
        for xp in xpath:
            elements = _parse_xml_file(filepath, xp, remove_tail)
            if elements:
                element = elements[0]  # take only the first match as element of the pair
            else:
                # if one of the twin examples cannot be found, skip it altogether rather than having incomplete twin examples
                return []
            text = _extract_text_from_elements([element], punkt, keep_xml, min_length=min_length)
            text = _cleanup(text)
            if text:
                text = text[0]  # _cleanup returns a list!
                # text = f"{xp}_{text}"  # debug!
                twin_examples.append(text)
            else:
                return []
        # from the multiple twin examples, generate a single concatenated with config.twin_delimiter as delimiter
        examples = [config.twin_delimiter.join(twin_examples)]
    # we could save the examples right here... might be faster with async
    n = 0
    for ex in examples:
        if ex:
            n += _save_task(ex, str(dest_file_path))
    return n


# switchable celery async task
if config.asynchr:
    examples_from_file_task = app.task(examples_from_file_task)


def _parse_xml_file(filepath: str, xpath: str, remove_tail: bool) -> List[str]:
    filepath = Path(filepath)
    with filepath.open() as f:
        xml = parse(f)
        elements = xml.xpath(xpath)
        if remove_tail:
            for e in elements:
                if e.tail is not None:
                    e.tail = None
    return elements


def _extract_text_from_elements(elements: Element, punkt: bool, keep_xml: bool, min_length: int) -> List[str]:
    examples = []
    if keep_xml:
        for e in elements:
            xml_str = tostring(e).decode('utf-8')  # tostring returns bytes
            text = innertext(e)
            if _filter(text, min_length):
                examples.append(xml_str)
    else:
        for e in elements:
            text = innertext(e)
            if punkt:
                sentences = PunktSentenceTokenizer().tokenize(text=text)
                filtered_sentences = [s for s in sentences if _filter(s, min_length)]
                examples += filtered_sentences
            else:
                if _filter(text, min_length):
                    examples.append(text)
    return examples


def _cleanup(examples: List[str]) -> List[str]:
    examples = [cleanup(e) for e in examples]
    return examples


def _filter(example: str, min_length: int) -> str:
    example = example if len(example) > min_length else ''
    return example


# @app.task
def _save_task(text: str, filepath: str,):
    """Writes each text on 1 line at the end of the file.
    Strips text from newlines so that it can be written on a single line.

    Args:
        text (str): the text, will be stripped of newline
        filepath (Path): the path to the file
    """
    with Path(filepath).open('a', encoding='utf-8') as f:  # mode 'a' to append lines
        f.write(f"{text.strip()}\n")
    return 1


# if config.asynchr:
#     save_task = app.task(save_task)


class ExtractorXML:
    """Extract multiple text examples from xml documents based on an XPath selector.
    Examples are appended as lines in a single file.
    The XPath specifies which element to extract from each xml file.
    By default, the inner text from the selected element will be saved as an example.
    It is also possible to prevent this and keep the xml markup, which is useful to train token classification tasks.
    If sentence level tokenization is set to True, the text is first split into sentences which are individually saved.

    Args:
        corpus (str):
            The directory of the corpus of source documents
        destination_dir (str, default to ""):
            Directory for the desitnation file (one line per example).
        sentence_level (bool, default to False):
            Whether to split the innert text into sentences, which will be saved as individual examples.
        xpath (str, default to ".//abstract"):
            The XPath expression to selecte the XML element from which the innertext will be extracted
        keep_xml (bool, default to False):
            Whether to keep the XML string instead of extracting the innertext
        remove_tail (bool, default to True):
                Whether to remove the tail of the xml selected xml element.
        inclusion_probability (float, default to 1.0):
            Probability of inclusuion of individiaul examples into the dataset; allows to take random subsample of very large dataset.
        subsets (List[str], default to ["train", "eval", "test"]):
            The names of the three subdirectories for train, eval and test sets
    """

    ALLOWED_EXTENSION = ['.xml', '.XML', '.nxml']

    def __init__(
        self,
        corpus: str,
        destination_dir: str = '',
        sentence_level: bool = False,
        xpath: Union[str, List[str]] = ".//abstract",
        keep_xml: bool = False,
        remove_tail: bool = True,
        inclusion_probability: float = 1.0,
        subsets: List[str] = ["train", "eval", "test"]
    ):
        self.corpus = Path(corpus)
        self.destination_dir = destination_dir
        self.sentence_level = sentence_level
        self.xpath = xpath
        self.keep_xml = keep_xml
        self.remove_tail = remove_tail
        self.inclusion_probability = inclusion_probability
        self.subsets = subsets
        if not self.destination_dir:
            basename = self.corpus.name
            self.destination_dir = Path("/data/text") / basename
        else:
            self.destination_dir = Path(self.destination_dir)
        if self.destination_dir.exists():
            raise ValueError(f"{self.destination_dir} already exists! Will not overwrite pre-existing dataset.")
        elif not self.destination_dir.parents[0].exists():
            raise ValueError(f"{self.destination_dir.parents[0]} does not exist, cannot proceed")
        else:
            self.destination_dir.mkdir()
            print(f"{self.destination_dir} created")
        self.source_dir_paths = [self.corpus / subset for subset in self.subsets]
        self.destination_file_paths = [self.destination_dir / f"{subset}.txt" for subset in subsets]
        if any([p.exists() for p in self.destination_file_paths]):
            raise ValueError(f"{', '.join([str(p) for p in self.destination_file_paths])} already exist. Cannot proceed.")
        else:
            if not all([source.exists() for source in self.source_dir_paths]):
                raise ValueError(f"The source {self.corpus} must include {' & '.join(subsets)} sub-directories. Cannot proceed.")

    def extract_from_corpus(self) -> int:
        """Method to extract example from corpus

        Returns:
            (Dict[str, int]):
                A dictionary with the number of examples saved to disk for each subset.
        """
        saved_num = {}
        for source_dir_path, destination_file_path in zip(self.source_dir_paths, self.destination_file_paths):
            N = self._run(
                source_dir_path,
                destination_file_path,
                self.xpath,
                sentence_level=self.sentence_level,
                keep_xml=self.keep_xml,
                remove_tail=self.remove_tail,
                inclusion_probability=self.inclusion_probability
            )
            saved_num[destination_file_path] = N
        return saved_num

    def _run(
        self,
        source_dir_path: Path,  # many files in the source dir
        dest_file_path: Path,  # one file as output with one line per example
        xpath: Union[str, List[str]],
        sentence_level: bool = False,
        keep_xml: bool = False,
        remove_tail: bool = True,
        inclusion_probability: float = 1.0,
        min_length: int = config.min_char_length
    ) -> int:

        num_saved_examples = 0
        batch_size = config.celery_batch_size
        filepaths = [f for f in source_dir_path.iterdir() if f.suffix in self.ALLOWED_EXTENSION]
        N = len(filepaths)
        for start in tqdm(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            if config.asynchr:  # non deterministic results order!
                task_list = [
                    examples_from_file_task.s(str(filepath), str(dest_file_path), xpath, sentence_level, keep_xml, remove_tail, min_length)
                    for indx, filepath in enumerate(filepaths[start:end])
                ]
                job = celery.group(task_list)
                results = job.apply()
                try:
                    results = results.get()
                except Error:
                    return []
                # remove empty sublists of examples
                results = [r for r in results if r]
                n = sum(results)
                num_saved_examples += n
            else:  # deterministic
                results = []
                for indx, filepath in enumerate(filepaths[start:end]):
                    n = examples_from_file_task(str(filepath), str(dest_file_path), xpath, sentence_level, keep_xml, remove_tail, min_length)
                    num_saved_examples += n
        return num_saved_examples

