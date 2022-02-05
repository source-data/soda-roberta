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
import datasets


class SourceDataNLP(datasets.GeneratorBasedBuilder):
    """SourceDataNLP provides datasets to train NLP tasks in cell and molecular biology."""

    _NER_LABEL_NAMES = [
        "O",
        "I-SMALL_MOLECULE",
        "B-SMALL_MOLECULE",
        "I-GENEPROD",
        "B-GENEPROD",
        "I-SUBCELLULAR",
        "B-SUBCELLULAR",
        "I-CELL",
        "B-CELL",
        "I-TISSUE",
        "B-TISSUE",
        "I-ORGANISM",
        "B-ORGANISM",
        "I-EXP_ASSAY",
        "B-EXP_ASSAY",
    ]
    _SEMANTIC_ROLES_LABEL_NAMES = ["O", "I-CONTROLLED_VAR", "B-CONTROLLED_VAR", "I-MEASURED_VAR", "B-MEASURED_VAR"]
    _BORING_LABEL_NAMES = ["O", "I-BORING", "B-BORING"]
    _PANEL_START_NAMES = ["O", "B-PANEL_START"]

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

    _HOMEPAGE = "https://huggingface.co/datasets/EMBO/sd-nlp"

    _LICENSE = "CC-BY 4.0"

    _URLS = {
        "NER": "https://huggingface.co/datasets/EMBO/sd-nlp/resolve/main/sd_panels.zip",
        "ROLES": "https://huggingface.co/datasets/EMBO/sd-nlp/resolve/main/sd_panels.zip",
        "BORING": "https://huggingface.co/datasets/EMBO/sd-nlp/resolve/main/sd_panels.zip",
        "PANELIZATION": "https://huggingface.co/datasets/EMBO/sd-nlp/resolve/main/sd_figs.zip",
    }

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="NER", version="0.0.1", description="Dataset for entity recognition"),
        datasets.BuilderConfig(name="ROLES", version="0.0.1", description="Dataset for semantic roles."),
        datasets.BuilderConfig(name="BORING", version="0.0.1", description="Dataset for semantic roles."),
        datasets.BuilderConfig(
            name="PANELIZATION",
            version="0.0.1",
            description="Dataset for figure legend segmentation into panel-specific legends.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "NER"

    def _info(self):
        if self.config.name == "NER":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._NER_LABEL_NAMES), names=self._NER_LABEL_NAMES)
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )
        elif self.config.name == "ROLES":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(self._SEMANTIC_ROLES_LABEL_NAMES), names=self._SEMANTIC_ROLES_LABEL_NAMES
                        )
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )
        elif self.config.name == "BORING":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._BORING_LABEL_NAMES), names=self._BORING_LABEL_NAMES)
                    ),
                }
            )
        elif self.config.name == "PANELIZATION":
            features = datasets.Features(
                {
                    "input_ids": datasets.Sequence(feature=datasets.Value("int32")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._PANEL_START_NAMES), names=self._PANEL_START_NAMES)
                    ),
                }
            )

        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features,
            supervised_keys=("input_ids", "labels"),
            homepage=self._HOMEPAGE,
            license=self._LICENSE,
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators.
        Uses local files if a data_dir is specified. Otherwise downloads the files from their official url."""
        if self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            url = _URLS[self.config.name]
            data_dir = dl_manager.download_and_extract(url)
            if self.config.name in ["NER", "ROLES", "BORING"]:
                data_dir += "/sd_panels"
            elif self.config.name == "PANELIZATION":
                data_dir += "/sd_figs"
            else:
                raise ValueError(f"unkonwn config name: {self.config.name}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir + "/train.jsonl",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir + "/test.jsonl",
                    "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir + "/eval.jsonl",
                    "split": "eval",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples. This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)"""

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "NER":
                    labels_type = data["label_ids"]["entity_types"]
                    tag_mask = [0 if tag == "O" else 1 for tag in labels_type]
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": labels_type,
                        "tag_mask": tag_mask
                    }
                elif self.config.name == "ROLES":
                    labels_type = data["label_ids"]["entity_types"]
                    geneprod = ["B-GENEPROD", "I-GENEPROD", "B-PROTEIN", "I-PROTEIN", "B-GENE", "I-GENE"]
                    tag_mask = [1 if t in geneprod else 0 for t in labels_type]
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["geneprod_roles"],
                        "tag_mask": tag_mask,
                    }
                elif self.config.name == "BORING":
                    yield id_, {"input_ids": data["input_ids"], "labels": data["label_ids"]["boring"]}
                elif self.config.name == "PANELIZATION":
                    yield id_, {
                        "input_ids": data["input_ids"],
                        "labels": data["label_ids"]["panel_start"],
                    }
