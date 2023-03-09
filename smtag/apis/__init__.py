from typing import Dict

import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
import os

load_dotenv()
FROM = os.getenv('FROM')

class Service:
    def __init__(self, REST_URL: str = "", HEADERS: Dict[str, str] = {}):
        self.REST_URL = REST_URL
        self.HEADERS = HEADERS
        self.retry_request = self.requests_retry_session()
        self.retry_request.headers.update(self.HEADERS)

    def requests_retry_session(
        self,
        retries=4,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        session=None
        ):
        """Creates a resilient session that will retry several times when a query fails.
        from  https://www.peterbe.com/plog/best-practice-with-retries-with-requests
        Parameters
        ----------
        retries : int, optional
            As in [`urllib3.util.Retry`](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry)
        backoff_factor : float, optional
            As in [`urllib3.util.Retry`](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry)
        status_forcelist : tuple, optional
            As in [`urllib3.util.Retry`](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry)
        session : requests.Session, optional
            If existing, a valid [`requests.Session` object](https://docs.python-requests.org/en/master/user/advanced/).
            If let to `None` it will create it.
            Usage:
            ```python
            session_retry = self.requests_retry_session()
            session_retry.headers.update({
                "Accept": "application/json",
                "From": "thomas.lemberger@embo.org"
            })
            response = session_retry.post(url, data=params, timeout=30)
            ```
        """
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
