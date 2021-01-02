import os
from dotenv import load_dotenv

load_dotenv()
LM_DATASET = os.getenv('LM_DATASET')
NER_DATASET = os.getenv('NER_DATASET')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
DEFAULT_TOKENIZER_NAME = os.getenv('DEFAULT_TOKENIZER_NAME')
LM_MODEL_PATH = os.getenv('LM_MODEL_PATH')
NER_MODEL_PATH = os.getenv('NER_MODEL_PATH')
HUGGINGFACE_CACHE = os.getenv('HUGGINGFACE_CACHE')