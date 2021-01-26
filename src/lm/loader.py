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


# template from : https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

"""Loading script for the biolang dataset for language modeling in biology."""

from __future__ import absolute_import, division, print_function

import json
from pathlib import Path
import datasets
from common import CACHE
import shutil

_CITATION = """\
@Unpublished{
    huggingface: dataset,
    title = {biolang},
    authors={Thomas Lemberger, EMBO},
    year={2021}
}
"""

_DESCRIPTION = """\
This dataset is based on abstracts from the open access section of PubMed Central to train language models for the domain of biology. 
"""

_HOMEPAGE = "https://europepmc.org/downloads/openaccess"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    'biolang': "...",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class BioLang(datasets.GeneratorBasedBuilder):
    """BioLang: a dataset to train language models in biology."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="MLM", version="0.0.1", description="Dataset for general masked language model."),
        datasets.BuilderConfig(name="DET", version="0.0.1", description="Dataset for part-of-speech (determinant) masked language model."),
        datasets.BuilderConfig(name="VERB", version="0.0.1", description="Dataset for part-of-speech (verbs) masked language model."),
        datasets.BuilderConfig(name="SMALL", version="0.0.1", description="Dataset for part-of-speech (determinants, conjunctions, prepositions, pronouns) masked language model."),
    ]

    DEFAULT_CONFIG_NAME = "MLM"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "MLM":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32"))
                }
            )
        elif self.config.name in ["DET", "VERB", "SMALL"]:
            features = datasets.Features({
                "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                "special_tokens_mask": datasets.Sequence(feature=datasets.Value("int8")),
            })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=('input_ids', 'pos_mask'),
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
                gen_kwargs={
                    "filepath": str(data_dir / "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": str(data_dir / "test.jsonl"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": str(data_dir / "eval.jsonl"),
                    "split": "eval",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "MLM":
                    yield id_, {
                        "input_ids": data["input_ids"],
                    }
                elif self.config.name == "DET":
                    pos_mask = [0] * len(data['input_ids'])
                    for idx, label in enumerate(data['label_ids']):
                        if label == 'DET':
                            pos_mask[idx] = 1
                    yield id_, {
                        "input_ids": data['input_ids'],
                        "tag_mask": pos_mask,
                        "special_tokens_mask": data['special_tokens_mask']
                    }
                elif self.config.name == "VERB":
                    pos_mask = [0] * len(data['input_ids'])
                    for idx, label in enumerate(data['label_ids']):
                        if label == 'VERB':
                            pos_mask[idx] = 1
                    yield id_, {
                        "input_ids": data['input_ids'],
                        "tag_mask": pos_mask,
                        "special_tokens_mask": data['special_tokens_mask']
                    }
                elif self.config.name == "SMALL":
                    pos_mask = [0] * len(data['input_ids'])
                    for idx, label in enumerate(data['label_ids']):
                        if label in ['DET', 'CCONJ', 'SCONJ', 'ADP', 'PRON']:
                            pos_mask[idx] = 1
                    yield id_, {
                        "input_ids": data['input_ids'],
                        "tag_mask": pos_mask,
                        "special_tokens_mask": data['special_tokens_mask']
                    }


def self_test():
    from datasets import load_dataset
    data_dir = "/tmp/dataset"
    p = Path(data_dir)
    p.mkdir()
    try:
        p_train = p / "train.jsonl"
        p_eval = p / "eval.jsonl"
        p_test = p / "test.jsonl"
        d = {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 0],
            "label_ids": ["X", "DET", "X", "X", "X", "X", "X", "X", "X", "X"],
            "special_tokens_mask": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        p_train.write_text(json.dumps(d))
        p_eval.write_text(json.dumps(d))
        p_test.write_text(json.dumps(d))
        train_dataset, eval_dataset, test_dataset = load_dataset(
            './lm/loader.py',
            'DET',
            data_dir=data_dir,
            split=["train", "validation", "test"],
            download_mode=datasets.utils.download_manager.GenerateMode.FORCE_REDOWNLOAD,
            cache_dir=CACHE
        )
        print(len(train_dataset))
        print(len(eval_dataset))
        print(len(test_dataset))

    finally:
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    self_test()
