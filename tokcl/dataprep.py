from pathlib import Path
from xml.etree.ElementTree import parse
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast, LineByLineTextDataset
from tokenizers import Encoding
from .encoder import featurize
from ..common.utils import innertext
from ..common.config import config
from ..common import TOKENIZER_PATH

tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, max_len=config.max_length)


class SDPreparator:
    """Processes source xml documents into examples that can be used in a token classification task.
    """
    def __init__(self, dir_path: Path, tokenizer):
        self.dir_path = dir_path
        self.filepaths = [p for p in dir_path.iterdir()]

    def encode_example(self, filepath: Path):
        with filepath.open() as f:
            xml = parse(f)
        features = featurize.encode(xml)
        features_th = tensorify(features, selected_features)
        inner_text = innertext(xml)
        tokenized = tokenizer(inner_text)
        labeled_token = align_codes(tokenized, features_th)
        self.save(labeled_token)

    def align_codes(self, tokenized: Encoding, features_th: torch.Tensor):
        for i in range(len(tokenized.tokens)):
            start, end = tokenized.offsets[i]
            codes = features_th[ : , start:end]
