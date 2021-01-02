import os
from dotenv import load_dotenv

load_dotenv()
SOURCE_XML_DIR = os.getenv('SOURCE_XML_DIR')
SOURCE_DATA_XML_DIR = os.getenv('SOURCE_DATA_XML_DIR')
TEXT_DIR = os.getenv('TEXT_DIR')
DATASET = os.getenv('DATASET')
NER_DATASET = os.getenv('NER_DATASET')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
DEFAULT_TOKENIZER_NAME = os.getenv('DEFAULT_TOKENIZER_NAME')
MODEL_PATH = os.getenv('MODEL_PATH')
HUGGINGFACE_CACHE = os.getenv('HUGGINGFACE_CACHE')