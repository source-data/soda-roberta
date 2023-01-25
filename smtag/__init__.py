import os
from dotenv import load_dotenv

load_dotenv()
LM_MODEL_PATH = os.getenv('LM_MODEL_PATH')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
TOKCL_MODEL_PATH = os.getenv('TOKCL_MODEL_PATH')
SEQ2SEQ_MODEL_PATH = os.getenv('SEQ2SEQ_MODEL_PATH')
CACHE = os.getenv('CACHE')
RUNS_DIR = os.getenv('RUNS_DIR')
