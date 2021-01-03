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
from .xmlcode import SourceDataCodes
import datasets
from common import NER_DATASET, HUGGINGFACE_CACHE


_NER_LABEL_NAMES = SourceDataCodes.ENTITY_TYPES.iob2_labels

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
    'entities': "",
    # 'roles': "",
    # 'panelization': ""
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
        # datasets.BuilderConfig(name="semantic_roles", version=0.1, description="Dataset for semantic roles."),
        # datasets.BuilderConfig(name="panelization", version=0.1, description="Dataset for figure legend segmentation into panel-specific legends."),
    ]

    DEFAULT_CONFIG_NAME = "NER"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "NER":  # This is the name of the configuration selected in BUILDER_CONFIGS above 
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
        # elif self.config.name == "semantic_roles":  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "tokens": datasets.Value("list_()"),
        #             "label_ids": datasets.Value("list_()"),
        #             "mask": datasets.Value("list_()")
        #         }
        #     )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("input_ids", "labels"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = Path(NER_DATASET)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "test.jsonl"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(data_dir / "eval.jsonl"),
                    "split": "eval",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "NER":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"],
                    }
                # else:
                #     yield id_, {
                #         "sentence": data["sentence"],
                #         "option2": data["option2"],
                #         "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                #     }


def self_test():
    from datasets import load_dataset
    train_dataset, eval_dataset, test_dataset = load_dataset(
        './tokcl/dataset.py',
        'NER',
        split=["train", "validation", "test"],
        download_mode=datasets.utils.download_manager.GenerateMode.FORCE_REDOWNLOAD,
        cache_dir=HUGGINGFACE_CACHE
    )
    print(len(train_dataset))
    print(len(eval_dataset))
    print(len(test_dataset))
    print(f"Number of classes: {train_dataset.info.features['labels'].feature.num_classes}")
    # train_10_80pct_ds = datasets.load_dataset('bookcorpus', split='train[:10%]+train[-80%:]')

if __name__ == "__main__":
    self_test()
