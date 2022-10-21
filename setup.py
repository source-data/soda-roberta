import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="smtag",
    version="3.0.0",
    python_requires='>=3.6',
    author="Source Data",
    author_email="source_data@embo.org",
    description="SmartTag provides methods to tag text from figure legends based on the SourceData model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/source-data/soda-roberta",
    packages=['smtag'],
    install_requires=[
        "torch",
        "tensorflow",
        "transformers",
        "datasets",
        "nltk",
        "scikit-learn",
        "python-dotenv",
        "seqeval",
        "celery",
        "flower",
        "lxml",
        "neo4j",
        "spacy",
        "notebook",
        "ipywidgets",
        "plotly",
    ],
    classifiers=(
        # full list: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries",
    ),
)
