# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import json
from pathlib import Path
from .xmlcode import SourceDataCodes as sd
import datasets
from transformers import RobertaTokenizerFast
from common import CACHE
from common.config import config
import shutil

_NER_LABEL_NAMES = sd.ENTITY_TYPES.iob2_labels
_SEMANTIC_ROLES_LABEL_NAMES = sd.GENEPROD_ROLES.iob2_labels
_BORING_LABEL_NAMES = sd.BORING.iob2_labels
_PANEL_START_NAMES = sd.PANELIZATION.iob2_labels
_GENEPROD = sd.GENEPROD.iob2_labels
_CELL_TYPE_LINE = sd.CELL_TYPE_LINE.iob2_labels

_CITATION = """\
@Unpublished{
    huggingface: dataset,
    title = {SourceData NLP},
    authors={Thomas Lemberger, EMBO},
    year={2021}
}
"""

_DESCRIPTION = """\
This dataset is based on the SourceData database and is intented to facilitate training of NLP tasks in the cell and molecualr biology domain. 
"""

_HOMEPAGE = "http://sourcedata.io"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    'NER': "",
    'ROLES': "",
    'PANELIZATION': ""
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class SourceDataNLP(datasets.GeneratorBasedBuilder):
    """SourceDataNLP provides datasets to train NLP tasks in cell and molecular biology."""

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="NER", version="0.0.1", description="Dataset for entity recognition"),
        datasets.BuilderConfig(name="CELL_TYPE_LINE", version="0.0.1", description="Dataset fortagging cell types and cell lines."),
        datasets.BuilderConfig(name="GENEPROD", version="0.0.1", description="Dataset for tagging geneproducts."),
        datasets.BuilderConfig(name="ROLES", version="0.0.1", description="Dataset for semantic roles."),
        datasets.BuilderConfig(name="BORING", version="0.0.1", description="Dataset for semantic roles."),
        datasets.BuilderConfig(name="PANELIZATION", version="0.0.1", description="Dataset for semantic roles."),
        # datasets.BuilderConfig(name="panelization", version=0.1, description="Dataset for figure legend segmentation into panel-specific legends."),
    ]

    DEFAULT_CONFIG_NAME = "NER"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "NER":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_NER_LABEL_NAMES),
                            names=_NER_LABEL_NAMES
                        )
                    ),
                }
            )
        elif self.config.name == "GENEPROD":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_GENEPROD),
                            names=_GENEPROD
                        )
                    ),
                }
            )
        elif self.config.name == "CELL_TYPE_LINE":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_CELL_TYPE_LINE),
                            names=_CELL_TYPE_LINE
                        )
                    ),
                }
            )
        elif self.config.name == "ROLES":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_SEMANTIC_ROLES_LABEL_NAMES),
                            names=_SEMANTIC_ROLES_LABEL_NAMES
                        )
                    ),
                }
            )
        elif self.config.name == "BORING":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_BORING_LABEL_NAMES),
                            names=_BORING_LABEL_NAMES
                        )
                    ),
                }
            )
        elif self.config.name == "PANELIZATION":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(_PANEL_START_NAMES),
                            names=_PANEL_START_NAMES
                        )
                    ),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=("input_ids", "labels"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = Path(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "train/data.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "test/data.jsonl"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "eval/data.jsonl"),
                    "split": "eval",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)"""

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "NER":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["entity_types"],
                    }
                elif self.config.name == "GENEPROD":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["geneprod"],
                    }
                elif self.config.name == "CELL_TYPE_LINE":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["cell_type_line"],
                    }
                elif self.config.name == "ROLES":
                    # masking of labeled entities to enforce learning from context
                    input_ids = data["input_ids"]
                    labels_type = data["label_ids"]["entity_types"]
                    labels_roles = data["label_ids"]["geneprod_roles"]
                    for i, t in enumerate(labels_type):
                        if t in ["B-GENEPROD", "I-GENEPROD", "B-PROTEIN", "I-PROTEIN", "B-GENE", "I-GENE"]:
                            input_ids[i] = self.tokenizer.mask_token_id
                    yield id_, {
                        "input_ids": input_ids,
                        "labels": labels_roles,
                    }
                elif self.config.name == "BORING":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["boring"]
                    }
                elif self.config.name == "PANELIZATION":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["panel_start"],
                    }


def self_test():
    from datasets import load_dataset
    data_dir = "/tmp/dataset"
    p = Path(data_dir)
    p.mkdir()
    try:
        p_train = p / "train"
        p_train.mkdir()
        p_train = p_train / "data.jsonl"
        p_eval = p / "eval"
        p_eval.mkdir()
        p_eval = p_eval / "data.jsonl"
        p_test = p / "test"
        p_test.mkdir()
        p_test = p_test / "data.jsonl"
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', max_len=config.max_length)
        batch_encoding = tokenizer("One two three four five six seven eight nine ten")
        d = {
            "input_ids": batch_encoding.input_ids,
            "label_ids": {
                "entity_types": ["O", "O", "O", "B-GENEPROD", "I-GENEPROD", "O", "O", "O", "O", "O", "O", "O"],
                "geneprod":  ["O", "O", "O", "B-GENEPROD", "I-GENEPROD", "O", "O", "O", "O", "O", "O", "O"],
                "cell_type_line":  ["O", "O", "O", "B-CELL", "I-CELL", "O", "O", "O", "O", "O", "O", "O"],
                "geneprod_roles": ["O", "O", "O", "B-CONTROLLED_VAR", "I-CONTROLLED_VAR", "O", "O", "O", "O", "O", "O", "O"],
                "boring": ["O", "O", "O", "B-BORING", "I-BORING", "O", "O", "O", "O", "O", "O", "O"],
                "panel_start": ["O", "B-PANEL_START", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
            },
        }
        p_train.write_text(json.dumps(d))
        p_eval.write_text(json.dumps(d))
        p_test.write_text(json.dumps(d))
        for configuration in ["NER", "ROLES", "BORING", "PANELIZATION", "CELL_TYPE_LINE", "GENEPROD"]:
            train_dataset, eval_dataset, test_dataset = load_dataset(
                './tokcl/loader.py',
                configuration,
                data_dir=data_dir,
                split=["train", "validation", "test"],
                download_mode=datasets.utils.download_manager.GenerateMode.FORCE_REDOWNLOAD,
                cache_dir=CACHE,
                tokenizer=tokenizer
            )
            print(len(train_dataset))
            print(len(eval_dataset))
            print(len(test_dataset))
            print(f"Number of classes: {train_dataset.info.features['labels'].feature.num_classes}")
    finally:
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    self_test()
