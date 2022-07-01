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

from __future__ import absolute_import, division, print_function

import json
import datasets

_BASE_URL = "https://huggingface.co/datasets/EMBO/sd-nlp-non-tokenized/resolve/main/"

class SourceDataNLP(datasets.GeneratorBasedBuilder):
    """SourceDataNLP provides datasets to train NLP tasks in cell and molecular biology."""

    _NER_LABEL_NAMES = [
        "O",
        "B-SMALL_MOLECULE",
        "I-SMALL_MOLECULE",
        "B-GENEPROD",
        "I-GENEPROD",
        "B-SUBCELLULAR",
        "I-SUBCELLULAR",
        "B-CELL",
        "I-CELL",
        "B-TISSUE",
        "I-TISSUE",
        "B-ORGANISM",
        "I-ORGANISM",
        "B-EXP_ASSAY",
        "I-EXP_ASSAY",
    ]
    _SEMANTIC_GENEPROD_ROLES_LABEL_NAMES =  ["O", "B-CONTROLLED_VAR", "I-CONTROLLED_VAR", "B-MEASURED_VAR", "I-MEASURED_VAR"]
    _SEMANTIC_SMALL_MOL_ROLES_LABEL_NAMES = ["O", "B-CONTROLLED_VAR", "I-CONTROLLED_VAR", "B-MEASURED_VAR", "I-MEASURED_VAR"]
    _BORING_LABEL_NAMES = ["O", "B-BORING", "I-BORING"]
    _PANEL_START_NAMES = ["O", "B-PANEL_START"]

    _CITATION = """\
    @Unpublished{
        huggingface: dataset,
        title = {SourceData NLP},
        authors={Thomas Lemberger & Jorge Abreu-Vicente, EMBO},
        year={2021}
    }
    """

    _DESCRIPTION = """\
    This dataset is based on the SourceData database and is intented to facilitate training of NLP tasks in the cell and molecualr biology domain.
    """

    _HOMEPAGE = "https://huggingface.co/datasets/EMBO/sd-nlp-non-tokenized"

    _LICENSE = "CC-BY 4.0"

    VERSION = datasets.Version("1.0.0")

    _URLS = {
        "NER": f"{_BASE_URL}sd_panels_general_tokenization.zip",
        "PANELIZATION": f"{_BASE_URL}sd_fig_general_tokenization.zip",
    }
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="NER", version=VERSION, description="Dataset for entity recognition"),
        datasets.BuilderConfig(name="GENEPROD_ROLES", version=VERSION, description="Dataset for semantic roles."),
        datasets.BuilderConfig(name="SMALL_MOL_ROLES", version=VERSION, description="Dataset for semantic roles."),
        datasets.BuilderConfig(name="BORING", version=VERSION, description="Dataset for semantic roles."),
        datasets.BuilderConfig(
            name="PANELIZATION",
            version=VERSION,
            description="Dataset for figure legend segmentation into panel-specific legends.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "NER"

    def _info(self):
        if self.config.name == "NER":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(feature=datasets.Value("string")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._NER_LABEL_NAMES),
                                                    names=self._NER_LABEL_NAMES)
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )
        elif self.config.name == "GENEPROD_ROLES":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(feature=datasets.Value("string")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(self._SEMANTIC_GENEPROD_ROLES_LABEL_NAMES),
                            names=self._SEMANTIC_GENEPROD_ROLES_LABEL_NAMES
                        )
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )
        elif self.config.name == "SMALL_MOL_ROLES":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(feature=datasets.Value("string")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(
                            num_classes=len(self._SEMANTIC_SMALL_MOL_ROLES_LABEL_NAMES),
                            names=self._SEMANTIC_SMALL_MOL_ROLES_LABEL_NAMES
                        )
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )
        elif self.config.name == "BORING":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(feature=datasets.Value("string")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._BORING_LABEL_NAMES),
                                                    names=self._BORING_LABEL_NAMES)
                    ),
                }
            )
        elif self.config.name == "PANELIZATION":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(feature=datasets.Value("string")),
                    "labels": datasets.Sequence(
                        feature=datasets.ClassLabel(num_classes=len(self._PANEL_START_NAMES),
                                                    names=self._PANEL_START_NAMES)
                    ),
                    "tag_mask": datasets.Sequence(feature=datasets.Value("int8")),
                }
            )

        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features,
            supervised_keys=("words", "label_ids"),
            homepage=self._HOMEPAGE,
            license=self._LICENSE,
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators.
        Uses local files if a data_dir is specified. Otherwise downloads the files from their official url."""

        if self.config.name in ["NER", "GENEPROD_ROLES", "SMALL_MOL_ROLES", "BORING"]:
            url = self._URLS["NER"]
            data_dir = dl_manager.download_and_extract(url)
            data_dir += "/sd_panels_general_tokenization"
        elif self.config.name == "PANELIZATION":
            url = self._URLS[self.config.name]
            data_dir = dl_manager.download_and_extract(url)
            data_dir += "/sd_fig_general_tokenization"
        else:
            raise ValueError(f"unkonwn config name: {self.config.name}")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir + "/train.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir + "/test.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir + "/eval.jsonl"},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples. This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)"""

        with open(filepath, encoding="utf-8") as f:
            # logger.info("⏳ Generating examples from = %s", filepath)
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "NER":
                    labels = data["label_ids"]["entity_types"]
                    tag_mask = [0 if tag == "O" else 1 for tag in labels]
                    yield id_, {
                        "words": data["words"],
                        "labels": labels,
                        "tag_mask": tag_mask
                    }
                elif self.config.name == "GENEPROD_ROLES":
                    labels = data["label_ids"]["geneprod_roles"]
                    geneprod = ["B-GENEPROD", "I-GENEPROD", "B-PROTEIN", "I-PROTEIN", "B-GENE", "I-GENE"]
                    tag_mask = [1 if t in geneprod else 0 for t in labels]
                    yield id_, {
                        "words": data["words"],
                        "labels": labels,
                        "tag_mask": tag_mask,
                    }
                elif self.config.name == "SMALL_MOL_ROLES":
                    labels = data["label_ids"]["small_mol_roles"]
                    small_mol = ["B-SMALL_MOLECULE", "I-SMALL_MOLECULE"]
                    tag_mask = [1 if t in small_mol else 0 for t in labels]
                    yield id_, {
                        "words": data["words"],
                        "labels": labels,
                        "tag_mask": tag_mask,
                    }
                elif self.config.name == "BORING":
                    yield id_, {"words": data["words"],
                                "labels": data["label_ids"]["boring"]}
                elif self.config.name == "PANELIZATION":
                    labels = data["label_ids"]["panel_start"]
                    tag_mask = [1 if t == "B-PANEL_START" else 0 for t in labels]
                    yield id_, {
                        "words": data["words"],
                        "labels": data["label_ids"]["panel_start"],
                        "tag_mask": tag_mask,
                    }

