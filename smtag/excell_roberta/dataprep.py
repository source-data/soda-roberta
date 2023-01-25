from pathlib import Path
from typing import List, Tuple, Dict
from lxml.etree import fromstring, Element
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding, AutoModelForTokenClassification
from smtag.xml2labels import CodeMap
from smtag.encoder import XMLEncoder
from smtag.utils import innertext
from datasets import Dataset, DatasetDict
from smtag.xml2labels import SourceDataCodes as sd
class PreparatorTOKCL:
    """Processes source xml documents into examples that can be used in a token classification task.
    It tokenizes the text with the provided tokenizer.
    The XML is used to generate labels according to the provided CodeMap.
    The datset is then split into train, eval, and test set and saved into json line files.

    Args:
        source_file_path (Path):
            The path to the source file.
        dest_file_path (Path):
            The path of the destination file where the files with the encoded labeled examples should be saved.
        tokenizer (RobertaTokenizerFast):
            The pre-trained tokenizer to use for processing the inner text.
        code_maps (List[CodeMap)]:
            A list of CodeMap, each specifying Tthe XML-to-code mapping of label codes to specific combinations of tag name and attribute values.
        max_length (int):
            Maximum number of token in one example. Examples will be truncated.
    """
    def __init__(
        self,
        source_dir_path: str,
        code_map: CodeMap,
        tokenizer: str ,
        subsets: List[str] = ["train", "eval", "test"],
        max_length: int = 514
    ):
        self.source_dir_path = Path(source_dir_path)
        self.subsets = subsets
        self.code_map = code_map
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_prefix_space=True)


    def run(self):
        """Runs the coding and labeling of xml examples.
        Saves the resulting text files to the destination directory.
        """

        data = {}

        for subset in self.subsets:
            print(f"Preparing: {subset}")
            source_file_path = self.source_dir_path / f"{subset}.txt"
            input_ids, labels, attention_mask = [], [], []
            with source_file_path.open() as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    xml_example: Element = fromstring(line)
                    input_ids_ex, labels_ex, attention_mask_ex = self._encode_example(xml_example)
                    input_ids.append(input_ids_ex)
                    labels.append(labels_ex)
                    attention_mask.append(attention_mask_ex)

            data[subset] = Dataset.from_dict({
                "input_ids": input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            })

        # We need to make sure this returns a dataset dictionary 

        return DatasetDict({
            "train": data["train"],
            "validation": data["eval"],
            "test": data["test"],
        })

    def _encode_example(self, xml: Element) -> Tuple[BatchEncoding, Dict]:
        xml_encoder = XMLEncoder(xml)
        inner_text = innertext(xml_encoder.element)
     
        if self.code_map.name == "entity_types":
            xml_encoded = xml_encoder.encode(self.code_map)

            char_level_labels = xml_encoded['label_ids']
            input_ids, labels, attention_mask = self._from_char_to_token_level_labels_roberta(inner_text,
                                                                            char_level_labels)
        elif self.code_map.name == "panel_start":
            xml_encoded = xml_encoder.encode(self.code_map)
            char_level_labels = [0] * len(xml_encoded['label_ids'])
            offsets = xml_encoded["offsets"]
            for offset in offsets:
                char_level_labels[offset[0]] = 1

            input_ids, labels, attention_mask = self._from_char_to_token_level_labels_panel_roberta(inner_text,
                                                                                    char_level_labels)
        else:
            xml_encoded = xml_encoder.encode(sd.ENTITY_TYPES)
            char_level_entities = xml_encoded['label_ids']
            xml_encoded = xml_encoder.encode(self.code_map)
            char_level_roles = xml_encoded['label_ids']
            input_ids, labels, attention_mask = self._from_char_to_token_level_labels_roles_roberta(inner_text,
                                                                                    char_level_entities,
                                                                                    char_level_roles)
        return input_ids, labels, attention_mask

    def _from_char_to_token_level_labels_roles_roberta(self, text: str, labels: List) -> List: # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying Tthe XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            text List[str]:     List of the characters inside the text of the XML elements
            labels List:        List of labels for each character inside the XML elements. They will be
                                a mix of integers and None

        Returns:
            List[str]           Word-level tokenized labels for the input text
        """

        # labels_ = [0 if label == None else label for label in labels]
            
        # tokens = self.tokenizer(text, is_split_into_words=False, truncation=True, max_length=self.max_length)
        
        # new_labels = []
        # for i in range(len(tokens["input_ids"])):
        #     if tokens.token_to_chars(i):
        #         token_char_map = labels_[tokens.token_to_chars(i).start: tokens.token_to_chars(i).end]
        #         if  token_char_map != []:
        #             if all(x==token_char_map[0] for x in token_char_map):
        #                 new_labels.append(token_char_map[0])
        #             else:
        #                 token_char_map_composed = [value for value in token_char_map if value != 0]
        #                 new_labels.append(token_char_map_composed[0])
        #         else:
        #             new_labels.append(0)
        #     else:
        #         new_labels.append(-100)

        # new_labels = self._labels_to_iob2_roberta(new_labels, tokens["input_ids"])

        # assert len(tokens["input_ids"]) == len(new_labels) == len(tokens["attention_mask"]), f"Length of labels and words not identical!"

        # return tokens["input_ids"], new_labels, tokens["attention_mask"]
        return (None, None, None)


    def _from_char_to_token_level_labels_roberta(self, text: str, labels: List) -> List: # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying Tthe XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            text List[str]:     List of the characters inside the text of the XML elements
            labels List:        List of labels for each character inside the XML elements. They will be
                                a mix of integers and None

        Returns:
            List[str]           Word-level tokenized labels for the input text
        """

        labels_ = [0 if label == None else label for label in labels]
            
        tokens = self.tokenizer(text, is_split_into_words=False, truncation=True, max_length=self.max_length)
        
        new_labels = []
        for i in range(len(tokens["input_ids"])):
            if tokens.token_to_chars(i):
                token_char_map = labels_[tokens.token_to_chars(i).start: tokens.token_to_chars(i).end]
                if  token_char_map != []:
                    if all(x==token_char_map[0] for x in token_char_map):
                        new_labels.append(token_char_map[0])
                    else:
                        token_char_map_composed = [value for value in token_char_map if value != 0]
                        new_labels.append(token_char_map_composed[0])
                else:
                    new_labels.append(0)
            else:
                new_labels.append(-100)

        new_labels = self._labels_to_iob2_roberta(new_labels, tokens["input_ids"])

        assert len(tokens["input_ids"]) == len(new_labels) == len(tokens["attention_mask"]), f"Length of labels and words not identical!"

        return tokens["input_ids"], new_labels, tokens["attention_mask"]
        
    def _from_char_to_token_level_labels_panel_roberta(self, text, labels):    
        
        tokens = self.tokenizer(text, is_split_into_words=False, truncation=True, max_length=self.max_length)
        
        new_labels = []

        for i in range(len(tokens["input_ids"])):
            if tokens.token_to_chars(i):
                token_char_map = labels[tokens.token_to_chars(i).start: tokens.token_to_chars(i).end]
                if 1 in token_char_map:
                    new_labels.append(1)
                else:
                    new_labels.append(0)
            else:
                new_labels.append(0)

        assert len(tokens["input_ids"]) == len(new_labels) == len(tokens["attention_mask"]), f"Length of labels and words not identical!"
        return tokens["input_ids"], new_labels, tokens["attention_mask"]



    def _labels_to_iob2_roberta(self, labels: List[int], tokens: List[int]): # Checked
        """
        Args:
            code_map (CodeMap): CodeMap, each specifying The XML-to-code mapping of label codes
                                to specific combinations of tag name and attribute values.
            text List[str]:     List of separated words
            labels List:        List of labels for each word inside the XML elements.

        Returns:
            List[str]           Word-level tokenized labels in IOB2 format

        """
        iob2_labels = []

        for idx, (label, token) in enumerate(zip(labels, tokens)):
            if label == -100:
                iob2_labels.append(-100)
            else:
                if label == 0:
                    iob2_labels.append(0)
                else:
                    if idx == 0:
                        iob2_labels.append(int(label) * 2 - 1)
                    if (idx > 0) and (labels[idx - 1] != label):
                        iob2_labels.append(int(label) * 2 - 1)
                    if (idx > 0) and (labels[idx - 1] == label):
                        iob2_labels.append(int(label) * 2)

        return iob2_labels

