from smtag.apis import Service, FROM
from typing import Dict, List
from requests.models import Response

class Uniprot(Service):

    def __init__(self):
        self.REST_URL = 'https://www.ebi.ac.uk/proteins/api/proteins'
        self.HEADERS = {
            "From": FROM,
            "Accept": "application/json"
        }
        Service.__init__(self, self.REST_URL, self.HEADERS)
    
    def _search(self, accession: str, reviewed: str = "true")  -> Response:

        # params = {
        #     'accession': accession,
        #     'reviewed': 'true',
        # }
        request_url = f'{self.REST_URL}?accession={accession}&reviewed={reviewed}'
        print(request_url)
        response = self.retry_request.get(request_url, headers=self.HEADERS, timeout=30)  # EuropePMC accepts only POST
        return response

    def get_recommended_names(self, accession: str) -> List[str]:
        recommended_names = []
        response = self._search(accession)
        print(response.status_code)
        if response.status_code == 200:
            response_json = response.json()[0] if isinstance(response.json(), list) else response.json()
            protein = response_json['protein']
            recommended_names.append(protein['recommendedName']['fullName']['value'])
            for alternative_name in protein['alternativeName']:
                recommended_names.append(alternative_name['fullName']['value'])
        return recommended_names

    def get_short_names(self, accession: str) -> List[str]:
        short_names = []
        response = self._search(accession)
        print(response.status_code)
        if response.status_code == 200:
            response_json = response.json()[0] if isinstance(response.json(), list) else response.json()
            protein = response_json['protein']
            for short_name in protein['recommendedName']['shortName']:
                short_names.append(short_name['value'])
            for alternative_name in protein['alternativeName']:
                for short_name in alternative_name.get('shortName', []):
                    short_names.append(short_name['value'])
        return short_names


if __name__ == '__main__':
    uniprot = Uniprot()
    accession = 'P34152'
    recommended_names = uniprot.get_recommended_names(accession)
    short_names = uniprot.get_short_names(accession)
    print(recommended_names)
    print(short_names)
