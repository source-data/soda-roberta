import os
from pathlib import Path
from dotenv import load_dotenv
from sdneo.db import Instance

load_dotenv()
SD_API_URL = os.getenv("SD_API_URL")
SD_API_USERNAME = os.getenv("SD_API_USERNAME")
SD_API_PASSWORD = os.getenv("SD_API_PASSWORD")

def get_db():
    NEO_URI = os.getenv('NEO_URI')
    NEO_USERNAME = os.getenv("NEO_USERNAME")
    NEO_PASSWORD = os.getenv("NEO_PASSWORD")
    return Instance(NEO_URI, NEO_USERNAME, NEO_PASSWORD)