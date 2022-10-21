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
        "torch==1.7.1",
        "tensorflow==2.4.0",
        "transformers==4.15.0",
        "datasets==1.2.1",
        "nltk==3.5",
        "scikit-learn==0.24.0",
        "python-dotenv==0.15.0",
        "seqeval==1.2.2",
        "celery",
        "flower",
        "lxml==4.6.2",
        "neo4j==4.1.1",
        "spacy==2.3.5",
        "notebook",
        "ipywidgets",
        "plotly",
    ],
)
