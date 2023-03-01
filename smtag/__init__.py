import os
from dotenv import load_dotenv
from smtag.sdneo.db import Instance

load_dotenv()
LM_MODEL_PATH = os.getenv('LM_MODEL_PATH')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
TOKCL_MODEL_PATH = os.getenv('TOKCL_MODEL_PATH')
SEQ2SEQ_MODEL_PATH = os.getenv('SEQ2SEQ_MODEL_PATH')
CACHE = os.getenv('CACHE')
RUNS_DIR = os.getenv('RUNS_DIR')
NEO_URI = os.getenv('NEO_URI')
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")
DB = Instance(NEO_URI, NEO_USERNAME, NEO_PASSWORD)