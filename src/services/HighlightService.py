import json
from typing import List, Dict
from urllib.request import Request, urlopen
from urllib.error import URLError

class HighlightService:
    """Service for interacting with the highlighting microservice."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def highlight(self, user_query: str, assertion: str, source_passages: List[str]) -> Dict[str, List[str]]:
        """
        Send a request to the highlighting microservice and return the highlighted sentences.

        Args:
            user_query (str): The user's query.
            assertion (str): The assertion to be checked.
            source_passages (List[str]): List of source passages to be analyzed.

        Returns:
            Dict[str, List[str]]: A dictionary containing the highlighted sentences.
        """
        url = f"{self.base_url}/highlight"
        
        payload = json.dumps({
            "user_query": user_query,
            "assertion": assertion,
            "source_passages": source_passages
        }).encode('utf-8')

        headers = {
            'Content-Type': 'application/json'
        }

        req = Request(url, data=payload, headers=headers, method='POST')

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf-8'))
        except URLError as e:
            raise Exception(f"Error communicating with highlighting service: {str(e)}")
