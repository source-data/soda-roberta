from pathlib import Path
from typing import List
from random import random
import celery
from tqdm import tqdm
from .celery_tasks import examples_from_file_task, save_task
from .config import config


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
        xpath: str = ".//abstract",
        keep_xml: bool = False,
        remove_tail: bool = True,
        inclusion_probability: float = 1.0,
        subsets: List[str] = ["train", "eval", "test"]
    ):
        self.corpus = Path(corpus)
        self.destination_dir = Path(destination_dir)
        self.sentence_level = sentence_level
        self.xpath = xpath
        self.keep_xml = keep_xml
        self.remove_tail = remove_tail
        self.inclusion_probability = inclusion_probability
        self.subsets = subsets
        if self.destination_dir:
            if not self.destination_dir.parents[0].exists():
                raise ValueError(f"{self.destination_dir.parents[0]} does not exists, cannot proceed")
            else:
                Path.mkdir(self.destination_dir, exist_ok=True)
                print(f"{self.destination_dir} created")
        else:
            basename = self.corpus.name
            self.destination_dir = Path("/data/text") / basename
        self.source_dir_paths = [self.corpus / subset for subset in self.subsets]
        self.destination_file_paths = [self.destination_dir / f"{subset}.txt" for subset in subsets]
        if any([True if p.exists() else False for p in self.destination_file_paths]):
            raise ValueError(f"{', '.join([str(p) for p in self.destination_file_paths])} already exist. Cannot proceed.")
        else:
            if not all([True if source.exists() else False for source in self.source_dir_paths]):
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
        selector: str,
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
            task_list = [
                examples_from_file_task.s(str(filepath), selector, sentence_level, keep_xml, remove_tail, min_length)
                for filepath in filepaths[start:end]
            ]
            job = celery.group(task_list)
            results = job.apply_async()
            results = results.get()
            # save to disk as we go
            saving_tasks = []
            for new_examples in results:
                for j, example in enumerate(new_examples):
                    proba = random()
                    if proba <= inclusion_probability:
                        saving_tasks.append(save_task.s(example, str(dest_file_path)))
            job = celery.group(saving_tasks)
            saving_results = job.apply_async()
            saving_results.get()
            num_saved_examples += len(saving_results)
        return num_saved_examples
