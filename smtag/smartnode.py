import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Union
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from lxml.etree import (
    Element, ElementTree,
    XMLParser, parse, XMLSyntaxError,
    fromstring, tostring
)
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm

from sdneo.db import Instance, Query

load_dotenv()
SD_API_URL = os.getenv("SD_API_URL")
SD_API_USERNAME = os.getenv("SD_API_USERNAME")
SD_API_PASSWORD = os.getenv("SD_API_PASSWORD")

try:
    # central logging facility in sd-graph
    import common.logging
    common.logging.configure_logging()
    logger = common.logging.get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")


class ResilientRequests:

    def __init__(self, user=None, password=None):
        self.session_retry = self.requests_retry_session()
        if user is not None and password is not None:
            self.session_retry.auth = (user, password)
        self.session_retry.headers.update({
            "Accept": "application/json",
            "From": "thomas.lemberger@embo.org"
        })

    @staticmethod
    def requests_retry_session(
        retries=4,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        session=None,
    ):
        # from  https://www.peterbe.com/plog/best-practice-with-retries-with-requests
        session = session if session is not None else requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def request(self, url: str, params: Dict = None) -> Dict:
        data = {}
        try:
            response = self.session_retry.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, str):
                    logger.error(f"skipping {url}: response is string and not json data: '''{data}'''")
                    data = {}
            else:
                logger.debug(f"failed loading json object with {url} ({response.status_code})")
        except Exception as e:
            logger.error("server query failed")
            logger.error(e)
        finally:
            if not data:
                logger.debug(f"data from {url} remains empty")
            return data


def doi2filename(doi: str) -> str:
    return doi.replace("/", "_").replace(".", "-")


def inner_text(xml_element: Element) -> str:
    if xml_element is not None:
        return "".join([t for t in xml_element.itertext()])
    else:
        return ""


@dataclass
class Properties:
    """Maps the SourceData REST API response fields to the properties of a SmartNode"""
    source: str = "sdapi"


@dataclass
class CollectionProperties:
    collection_name: str = ""
    collection_id: str = None

    def __str__(self):
        return f'"{self.collection_name}"'


class GET_COLLECTION(Query):

    code = '''
MATCH (coll:SDCollection {name: {$collection_name}}})
RETURN coll.name AS collection_name, coll.id AS collection_id
    '''
    returns = ['collection_name', 'collection_id']

class MERGE_COLLECTION(Query):

    code = '''
MERGE (coll:SDCollection)
WHERE
    coll.name = {$collection_name}
    coll.id = {$collection_id}
RETURN coll.name AS collection_name, coll.id AS collection_id
'''
    returns = ['collection_name', 'collection_id']


@dataclass
class ArticleProperties(Properties):
    doi: str = ""
    title: str = ""
    journal_name: str =""
    pub_date: str = ""
    pmid: str = ""
    pmcid: str = ""
    import_id: str = ""
    pub_year: str = "" # unfortunately SD has no pub_date properties
    nb_figures: int = 0

    def __str__(self):
        return f"\"{self.title}\" ({self.doi})"
    
class GET_ARTICLES(Query):

    code = '''
MATCH (article:SDArticle)
WHERE
    article.doi = {$doi}
RETURN
    article.doi AS doi, 
    article.title AS title,
    article.journal_name AS journal_name,
    article.pub_date AS pub_date,
    article.pmid AS pmid,
    article.pmcid AS pmcid,
    article.import_id AS import_id,
    article.pub_year AS pub_year,
    article.nb_figures AS nb_figures
    '''
    returns = ['doi', 'title', 'journal_name', 'pub_date', 'pmid', 'pmcid', 'import_id', 'pub_year', 'nb_figures']

class MERGE_ARTICLE(Query):

    code = '''
MERGE (article:SDArticle)
WHERE
    article.doi = {$doi}
SET
    article.title = {$title},
    article.journal_name = {$journal_name},
    article.pub_date = {$pub_date},
    article.pmid = {$pmid},
    article.pmcid = {$pmcid},
    article.import_id = {$import_id},
    article.pub_year = {$pub_year},
    article.nb_figures = {$nb_figures}
RETURN
    article.doi AS doi
    '''
    returns = ['doi']


@dataclass
class FigureProperties(Properties):
    paper_doi: str = ""
    figure_label: str = ""
    figure_id: str = ""
    figure_title: str = ""
    # caption: str = ""
    href: str = ""

    def __str__(self):
        return f"\"{self.figure_label}\" ({self.figure_id})"


@dataclass
class PanelProperties(Properties):
    paper_doi: str = ""
    figure_label: str = ""
    figure_id: str = ""
    panel_id: str = ""
    panel_label: str = ""
    panel_number: str = ""
    caption: str = ""
    formatted_caption: str = ""
    href: str = ""
    coords: str = ""

    def __str__(self):
        return f"\"{self.panel_number}\" ({self.panel_id})"


@dataclass
class TaggedEntityProperties(Properties):
    tag_id: str = ""
    category: str = ""
    entity_type: str = ""
    role: str = ""
    text: str = ""
    ext_ids: str = ""
    ext_dbs: str = ""
    in_caption: str = ""
    ext_names: str = ""
    ext_tax_ids: str = ""
    ext_tax_names: str = ""
    ext_urls: str = ""

    def __str__(self):
        return f"\"{self.text}\" ({', '.join(filter(lambda x: x is not None, [self.category, self.entity_type, self.role]))})"


class SourceDataAPIParser:
    """Parses the response of the SourceData REST API and maps the fields to the internal set of properties of SmartNodes"""

    @staticmethod
    def collection_props(response: List) -> CollectionProperties:
        if response:
            response = response[0]
        else:
            response = {}
        props = {
            "collection_name": response.get("name", ""),