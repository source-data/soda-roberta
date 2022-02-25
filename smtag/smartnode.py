from enum import auto
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from lxml.etree import Element, ElementTree, XMLParser, XMLSyntaxError, fromstring
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm

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


def doi2filename(doi: str):
    return doi.replace("/", "_").replace(".", "-")


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
            "collection_id": response.get("collection_id", ""),
        }
        return CollectionProperties(**props)

    @staticmethod
    def children_of_collection(response: List, collection_id: str) -> List["Article"]:
        article_ids = []
        logger.debug(f"collection {collection_id} has {len(response)} elements.")
        for article_summary in response:
            doi = article_summary.get("doi", "")
            sdid = article_summary.get("id", "")
            title = article_summary.get("title", "")
            collections = article_summary.get("collections", [])
            collection_names = [c["name"] for c in collections]
            # try to find an acceptable identifier
            if doi:
                article_ids.append(doi)
            elif sdid:
                logger.debug(f"using sdid {sdid} instead of doi for: \n{title}.")
                article_ids.append(sdid)
            else:
                logger.error(f"no doi and no sd id for {title} in collection {collection_names}.")
        # remove duplicates
        article_ids = list(set(article_ids))
        return article_ids

    @staticmethod
    def article_props(response: Dict) -> ArticleProperties:
        # {"title":"A non-death function of the mitochondrial apoptosis apparatus in immunity","year":"2019","pmcid":"SD4730","pmid":null,"doi":"10.15252/embj.2018100907","authors":"Dominik Brokatzky, Benedikt Dörflinger, Aladin Haimovici, Arnim Weber, Susanne Kirschnek, Juliane Vier, Arlena Metz, Julia Henschel, Tobias Steinfeldt, Ian, E. Gentle, Georg Häcker","journal":"A non-death function of the mitochondrial apoptosis apparatus in immunity","nbFigures":"4","tax_id":null,"taxon":null}
        nb_figures = int(response.get("nbFigures", 0))
        props = {
            "doi": response.get("doi", ""),
            "title": response.get("title", ""),
            "journal_name": response.get("journal", ""),
            "pub_date": response.get("pub_date", ""),
            "pmid": response.get("pmid", ""),
            "pmcid": response.get("pmcid", ""),
            "pub_year": response.get("year", ""),
            "nb_figures": nb_figures
        }
        return ArticleProperties(**props)

    def children_of_article(self, response: List, collection_id: str, doi: str) -> List["Figure"]:
        nb_figures = int(response.get("nbFigures", 0))
        fig_indices = range(1, nb_figures + 1)  # figures are 1-indexed
        return fig_indices

    @staticmethod
    def figure_props(response: Dict, doi: str) -> FigureProperties:
        # {"figure_id":"26788","label":"Figure 1","caption":"<p><strong>Figure 1</strong> ...</p>\n","panels":["72266","72258","72259","72260","72261","72262","72263","72264","72265"],"href":"https://api.sourcedata.io/file.php?figure_id=26788"}
        fig_title = response.get("fig_title", "")
        fig_caption = response.get("caption", "")
        if not fig_title and fig_caption:
            # strip caption of any HTML/XML tags
            cleaned_fig_caption = BeautifulSoup(fig_caption, 'html.parser').get_text()
            # from O'Reilly's Regular Expressions Cookbook
            # cleaned_fig_caption = re.sub(r'''</?([A-Za-z][^\s>/]*)(?:[^>"']|"[^"]*"|'[^']*')*>''', fig_caption, '')
            first_sentence = re.match(r"\W*([^\n\r]*?)[\.\r\n]", cleaned_fig_caption)
            if first_sentence:
                fig_title = first_sentence.group(1)
                fig_title = re.sub(r"fig[.\w\s]+\d", "", fig_title, flags=re.IGNORECASE)
                fig_title += "." # adds a dot just in case it is missing
                fig_title = fig_title.replace("..", ".") # makes sure that title finishes with a single . 
        props = {
            "paper_doi": doi,
            "figure_label": response.get("label", ""),
            "figure_id": response.get("figure_id", ""),
            "figure_title": fig_title,
            # "caption": fig_caption,
            "href": response.get("href", ""),
        }
        return FigureProperties(**props)

    def children_of_figures(self, response: List) -> List["Panel"]:
        # find the panel ids
        panel_ids = response.get("panels",[])
        return panel_ids

    @staticmethod
    def panel_props(response: Dict) -> PanelProperties:
        def cleanup(panel_caption: str):
            # need protection agains missing spaces after parenthesis, typically in figure or panel labels
            try:
                parenthesis = re.search(r'(\(.*?\))(\w)', panel_caption)
            except TypeError as e:
                print("#"*25)
                print(type(panel_caption))
                print(panel_caption)
                raise e

            if parenthesis:
                logger.debug("adding space after closing parenthesis {}".format(re.findall(r'(\(.*?\))(\w)', panel_caption)))
                panel_caption = re.sub(r'(\(.*?\))(\w)',r'\1 \2', panel_caption)
            # protection against carriage return
            if re.search('[\r\n]', panel_caption):
                logger.debug(f"removing return characters in {panel_caption}")
                panel_caption = re.sub('[\r\n]', '', panel_caption)
            # protection against <br> instead of <br/>
            panel_caption = re.sub(r'<br>', r'<br/>', panel_caption)
            # protection against badly formed link elements
            panel_caption = re.sub(r'<link href="(.*)">', r'<link href="\1"/>', panel_caption)
            panel_caption = re.sub(r'<link href="(.*)"/>(\n|.)*</link>', r'<link href="\1">\2</link>', panel_caption)
            # protection against missing <sd-panel> tags
            if re.search(r'^<sd-panel>(\n|.)*</sd-panel>$', panel_caption) is None:
                logger.debug(f"correcting missing <sd-panel> </sd-panel> tags in {panel_caption}")
                panel_caption = '<sd-panel>' + panel_caption + '</sd-panel>'
            # protection against nested or empty sd-panel
            panel_caption = re.sub(r'<sd-panel><sd-panel>', r'<sd-panel>', panel_caption)
            panel_caption = re.sub(r'</sd-panel></sd-panel>', r'</sd-panel>', panel_caption)
            panel_caption = re.sub(r'<sd-panel/>', '', panel_caption)
            # We may loose a space that separates panels in the actual figure legend...
            panel_caption = re.sub(r'</sd-panel>$', r' </sd-panel>', panel_caption)
            # and then remove possible runs of spaces
            panel_caption = re.sub(r' +', r' ', panel_caption)
            return panel_caption

        panel_id = response.get("current_panel_id", "")
        # the SD API panel method includes "reverse" info on source paper, figures, and all the other panels
        # take the portion of the data returned by the REST API that concerns panels
        paper_info = response.get("paper", {})
        figure_info = response.get("figure", {})
        panels = figure_info.get("panels", [])
        # transform into dict
        panels = {p["panel_id"]: p for p in panels}
        panel_info = panels.get(panel_id, {})
        paper_doi = paper_info.get("doi", "")
        figure_label = figure_info.get("label", "")
        figure_id = figure_info.get("figure_id")
        panel_id = panel_info.get("panel_id", "")  # "panel_id":"72258",
        panel_label = panel_info.get("label", "")  # "label":"Figure 1-B",
        panel_number = panel_info.get("panel_number", "")  # "panel_number":"1-B",
        caption = panel_info.get("caption", "")
        try:
            caption = cleanup(caption)
        except TypeError as e:
            print("#"*50)
            print(type(caption))
            print(caption)
            print(f"\n\n\n\n{response}\n\n\n\n")
            print("#"*50)
            import pdb; pdb.set_trace()
            raise e
        formatted_caption = panel_info.get("formatted_caption", "")
        href = panel_info.get("href", "")  # "href":"https:\/\/api.sourcedata.io\/file.php?panel_id=72258",
        coords = panel_info.get("coords", {})  # "coords":{"topleft_x":346,"topleft_y":95,"bottomright_x":632,"bottomright_y":478}
        coords = ", ".join([f"{k}={v}" for k, v in coords.items()])
        props = {
            "paper_doi": paper_doi,
            "figure_label": figure_label,
            "figure_id": figure_id,
            "panel_id": panel_id,
            "panel_label": panel_label,
            "panel_number": panel_number,
            "caption": caption,
            "formatted_caption": formatted_caption,
            "href": href,
            "coords": coords,
        }
        return PanelProperties(**props)

    def children_of_panels(self, response: List) -> List["TaggedEntity"]:
        panel_id = response.get("current_panel_id")
        panels = response.get("figure", {}).get("panels", [])
        # transform into dict
        panels = {p["panel_id"]: p for p in panels}
        current_panel = panels[panel_id]
        tags_data = current_panel.get("tags", [])
        return tags_data

    def tagged_entity_props(self, response: List) -> TaggedEntityProperties:
        tag_id = response.get("id", "")
        category = response.get("category", "entity")
        entity_type = response.get("type", "")
        role = response.get("role", "")
        text = response.get("text", "")
        ext_ids = "///".join(response.get("external_ids", []))
        ext_dbs = "///".join(response.get("externalresponsebases", []))
        in_caption = response.get("in_caption", "") == "Y"
        ext_names = "///".join(response.get("external_names", []))
        ext_tax_ids = "///".join(response.get("external_tax_ids", []))
        ext_tax_names = "///".join(response.get("external_tax_names", []))
        ext_urls = "///".join(response.get("external_urls", []))
        props = {
            "tag_id": tag_id,
            "category": category,
            "entity_type": entity_type,
            "role": role,
            "text": text,
            "ext_ids": ext_ids,
            "ext_dbs": ext_dbs,
            "in_caption": in_caption,
            "ext_names": ext_names,
            "ext_tax_ids": ext_tax_ids,
            "ext_tax_names": ext_tax_names,
            "ext_urls": ext_urls,
        }
        return TaggedEntityProperties(**props)


@dataclass
class Relationship:
    """Specifies the target of a directional typed relationship to another SmartNode """
    rel_type: str = ""
    target: "SmartNode" = None


class XMLSerializer:
    """Recursively serializes the properties of SmartNodes and of their descendents."""

    XML_Parser = XMLParser(recover=True)

    def generate_article(self, article: "Article") -> Element:
        xml_article = Element('article', doi=article.props.doi)
        xml_article = self.append_children_of_article(xml_article, article)
        return xml_article

    def append_children_of_article(self, xml_article: Element, article: "Article") -> Element:
        figures = [rel.target for rel in article.relationships if rel.rel_type == "has_figure"]
        xml_figures = [self.generate_figure(fig) for fig in figures]
        # do this here since there might be cases where several types of relationships have to be combined
        for xml_fig in xml_figures:
            xml_article.append(xml_fig)
        return xml_article

    def generate_figure(self, figure: "Figure") -> Element:
        xml_fig = Element('fig', id=figure.props.figure_id)
        xml_title = Element('title')
        xml_title.text = figure.props.figure_title
        xml_fig.append(xml_title)
        xml_fig_label = Element('label')
        xml_fig_label.text = figure.props.figure_label
        xml_fig.append(xml_fig_label)
        graphic_element = Element('graphic', href=figure.props.href)
        xml_fig.append(graphic_element)
        xml_fig = self.append_children_of_figure(xml_fig, figure)
        return xml_fig

    def append_children_of_figure(self, xml_fig: Element, figure: "Figure") -> Element:
        panels = [rel.target for rel in figure.relationships if rel.rel_type == "has_panel"]
        xml_panels = [self.generate_panel(panel) for panel in panels]
        for xml_panel in xml_panels:
            xml_fig.append(xml_panel)
        return xml_fig

    def generate_panel(self, panel: "Panel") -> Element:
        caption = panel.props.caption
        try:
            if caption:
                xml_panel = fromstring(caption, parser=self.XML_Parser)
            else:
                xml_panel = Element("sd-panel")
            xml_panel.attrib['panel_id'] = str(panel.props.panel_id)
            if panel.props.href:
                graphic_element = Element('graphic', href=panel.props.href)
                xml_panel.append(graphic_element)
        except XMLSyntaxError as err:
            n = int(re.search(r'column (\d+)', str(err)).group(1))
            logger.error(f"XMLSyntaxError: ```{caption[n-10:n]+'!!!'+caption[n]+'!!!'+caption[n+1:n+10]}```")
            xml_panel = None
        return xml_panel


class SmartNode:

    # NEO4J: Instance = DB
    SD_REST_API: str = "https://api.sourcedata.io/"
    REST_API_PARSER = SourceDataAPIParser()
    # SOURCE_XML_DIR: str = "xml_source_files/"
    DEST_XML_DIR: str = "xml_destination_files/"
    XML_SERIALIZER = XMLSerializer()

    def __init__(self, ephemeral: bool = False):
        self.props: Properties = None
        self._relationships: List[Relationship] = []
        self.ephemeral = ephemeral

    # def from_db(self, id: str, overwire: bool = True):
    #     """Instantiates self from the database """
    #     raise NotImplementedError

    # def to_db(self, mode: str = "MERGE"):
    #     """Updates or create node and its descendents in the database"""
    #     raise NotImplementedError

    # def from_xml(self, xml_source: str):
    #     """Loads and parses a jats file to instantiate properties and children"""
    #     raise NotImplementedError

    def to_xml(self, sub_dir: str) -> Element:
        """Serializes the object and its descendents as xml file"""
        raise NotImplementedError

    def from_sd_REST_API(self, id: str):
        """Instantiates properties and children from the SourceData REST API"""
        raise NotImplementedError

    @staticmethod
    def _request(url: str) -> Dict:
        response = ResilientRequests(SD_API_USERNAME, SD_API_PASSWORD).request(url)
        return response

    def _save_xml(self, xml_element: Element, sub_dir: str, basename: str) -> str:
        dest_dir = Path(self.DEST_XML_DIR)
        dest_dir.mkdir(exist_ok=True)
        dest_dir = dest_dir / sub_dir if sub_dir else dest_dir
        filename = basename + ".xml"
        filepath = dest_dir / filename
        if self.ephemeral and self.auto_save and not self.relationships:
            logger.warning(f"There are no relationships left in an ephermeral auto-saved object. Attempt to save to {str(filepath)} is likely to be a mistake.")
        if filepath.exists():
            logger.error(f"{filepath} already exists, not overwriting.")
        else:
            filepath = str(filepath)
            logger.info(f"writing to {filepath}")
            ElementTree(xml_element).write(filepath, encoding='utf-8', xml_declaration=True)
        return str(filepath)

    def _sync(self):
        pass
        """Synchronizes with database, whatever it means in terms of filling gaps and overwriting properties"""
        # first MERGE node with neo4j node
        # self.to_db()
        # then update properties of python object
        # self.from_db()

    @property
    def relationships(self) -> List[Relationship]:
        return self._relationships

    def _set_relationships(self, rel_type: str, targets: List["SmartNode"]):
        # keep _relationships as a list rather than a Dict[rel_type, nodes] in case staggered order is important
        self._relationships += [Relationship(rel_type=rel_type, target=target) for target in targets]

    def _finish(self) -> "SmartNode":
        self._sync()  # synchronize with the database
        if self.ephemeral:
            # reset relationships to free memory from descendants
            self._relationships = []
        return self

    def to_str(self, level=0):
        indentation = "  " * level
        s = ""
        s += indentation + f"{self.__class__.__name__} {self.props}\n"
        for rel in self.relationships:
            s += indentation + f"-[{rel.rel_type}]->\n"
            s += rel.target.to_str(level + 1) + "\n"
        return s

    def __str__(self):
        return self.to_str()


class Collection(SmartNode):

    GET_COLLECTION = "collection/"
    GET_LIST = "/papers"

    def __init__(self, *args, auto_save: bool = True, sub_dir: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_dir = sub_dir
        self.auto_save = auto_save
        self.props = CollectionProperties()

    def from_sd_REST_API(self, collection_name: str) -> SmartNode:
        logger.debug(f"from sd API collection {collection_name}")
        url_get_collection = self.SD_REST_API + self.GET_COLLECTION + collection_name
        response_1 = self._request(url_get_collection)
        if response_1:
            self.props = self.REST_API_PARSER.collection_props(response_1)
            url_get_list_of_papers = self.SD_REST_API + self.GET_COLLECTION + self.props.collection_id + self.GET_LIST
            response_2 = self._request(url_get_list_of_papers)
            article_ids = self.REST_API_PARSER.children_of_collection(response_2, self.props.collection_id)
            articles = []
            for article_id in tqdm(article_ids, desc="articles"):
                # if collection auto save is on, each article is saved as we go
                # if the collection is ephemeral, no point in keeping relationships in article after saving and article are ephemeral too
                article = Article(auto_save=self.auto_save, ephemeral=self.auto_save).from_sd_REST_API(self.props.collection_id, article_id)
                articles.append(article)
            self._set_relationships("has_article", articles)
        return self._finish()

    def to_xml(self, sub_dir: str = None) -> List[str]:
        filepaths = []
        # in auto save mode, the articles are saved as soon as they are created
        if self.auto_save:
            logger.warning(f"articles were saved already since auto_save mode is {self.auto_save}")
        else:
            sub_dir = sub_dir if sub_dir is not None else self.sub_dir
            for rel in self.relationships:
                if rel.rel_type == "has_article":
                    article = rel.target
                    filepath = article.to_xml(sub_dir)
                    filepaths.append(filepath)
        return filepaths


class Article(SmartNode):

    GET_COLLECTION = "collection/"
    GET_ARTICLE = "paper/"

    def __init__(self, *args, auto_save: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_save = auto_save
        self.props = ArticleProperties()

    def from_sd_REST_API(self, collection_id: str, doi: str) -> SmartNode:
        logger.debug(f"  from sd API article {doi}")
        url = self.SD_REST_API + self.GET_COLLECTION + collection_id + "/" + self.GET_ARTICLE + doi
        response = self._request(url)
        if response:
            self.props = self.REST_API_PARSER.article_props(response)
            fig_indices = self.REST_API_PARSER.children_of_article(response, collection_id, doi)
            figures = []
            for idx in tqdm(fig_indices, desc="figures ", leave=False):
                fig = Figure().from_sd_REST_API(collection_id, doi, idx)
                figures.append(fig)
            self._set_relationships("has_figure", figures)
        return self._finish()

    def _finish(self) -> "SmartNode":
        if self.auto_save:
            logger.info("auto saving")
            self.to_xml()
        return super()._finish()

    def to_xml(self, sub_dir: str = None) -> str:
        xml = self.XML_SERIALIZER.generate_article(self)
        basename = self.props.doi.replace("/", "_").replace(".", "-")
        filepath = self._save_xml(xml, sub_dir, basename)
        return filepath


class Figure(SmartNode):

    GET_COLLECTION = "collection/"
    GET_ARTICLE = "paper/"
    GET_FIGURE = "figure/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props = FigureProperties()

    def from_sd_REST_API(self, collection_id: str, doi: str, figure_index: str) -> SmartNode:
        logger.debug(f"    from sd API figure {figure_index}")
        url = self.SD_REST_API + self.GET_COLLECTION + collection_id + "/" + self.GET_ARTICLE + doi + "/" + self.GET_FIGURE + str(figure_index)
        response = self._request(url)
        if response:
            self.props = self.REST_API_PARSER.figure_props(response, doi)
            panel_ids = self.REST_API_PARSER.children_of_figures(response)
            panels = []
            for panel_id in tqdm(panel_ids, desc="panels  ", leave=False):
                panel = Panel().from_sd_REST_API(panel_id)
                panels.append(panel)
            self._set_relationships("has_panel", panels)
        return self._finish()

    def to_xml(self, sub_dir: str = None) -> str:
        xml = self.XML_SERIALIZER.generate_figure(self)
        doi = doi2filename(self.props.paper_doi)
        basename = "_".join([
            doi,
            "fig",
            self.props.figure_label,
            self.props.figure_id
        ])
        filepath = self._save_xml(xml, sub_dir, basename)
        return filepath


class Panel(SmartNode):

    GET_PANEL = "panel/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props = PanelProperties()

    def from_sd_REST_API(self, panel_id: str) -> SmartNode:
        logger.debug(f"      from sd API panel {panel_id}")
        url = self.SD_REST_API + self.GET_PANEL + panel_id
        response = self._request(url)
        if response:
            self.props = self.REST_API_PARSER.panel_props(response)
            tags_data = self.REST_API_PARSER.children_of_panels(response)
            tagged_entities = [TaggedEntity().from_sd_REST_API(tag)for tag in tags_data]
            self._set_relationships("has_entity", tagged_entities)
            self._sync()
        return self._finish()

    def to_xml(self, sub_dir: str = None) -> str:
        xml = self.XML_SERIALIZER.generate_panel(self)
        doi = doi2filename(self.props.paper_doi)
        basename = "_".join([
            doi,
            "fig",
            self.props.figure_label,
            self.props.panel_number,
            self.props.panel_id
        ])
        filepath = self._save_xml(xml, sub_dir, basename)
        return filepath


class TaggedEntity(SmartNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props = TaggedEntityProperties()

    def from_sd_REST_API(self, tag_data: List) -> SmartNode:
        logger.debug(f"        from sd tags {tag_data.get('text')}")
        self.props = self.REST_API_PARSER.tagged_entity_props(tag_data)
        return self._finish()
