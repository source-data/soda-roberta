from smtag.apis import Service, FROM
from typing import Dict, List
from requests.models import Response

class EPMC(Service):

    def __init__(self):
        self.REST_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
        self.HEADERS = {
            "From": FROM,
            "Content-type": "application/json;charset=UTF-8"
        }
        Service.__init__(self, self.REST_URL, self.HEADERS)
    
    def _search(self, query: str, limit: int = 1)  -> Response:
        article_list = []
        params = {
            'query': query,
            'resultType': 'core',
            'format': 'json',
            'pageSize': limit,
        }
        print(self.REST_URL, self.HEADERS, params)
        response = self.retry_request.get(self.REST_URL, params=params, headers=self.HEADERS, timeout=30)  # EuropePMC accepts only POST
        return response

    def get_abstract(self, doi: str) -> str:
        abstract = ''
        response = self._search(f"DOI:{doi}")
        if response.status_code == 200:
            response_json = response.json()
            if response_json['hitCount'] > 0:
                abstract = response_json['resultList']['result'][0]['abstractText']
        return abstract


if __name__ == '__main__':
    epmc = EPMC()
    doi = '10.1016/j.cell.2019.10.001'
    abstract = epmc.get_abstract(doi)
    print(abstract)
