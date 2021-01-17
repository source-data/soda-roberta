import os
from dotenv import load_dotenv

load_dotenv()
LM_DATASET = os.getenv('LM_DATASET')
LM_MODEL_PATH = os.getenv('LM_MODEL_PATH')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
TOKCL_DATASET = os.getenv('TOKCL_DATASET')
TOKCL_MODEL_PATH = os.getenv('TOKCL_MODEL_PATH')
CACHE = os.getenv('CACHE')