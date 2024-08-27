import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.models.data_models import AssertionModel


class HighlightService:
    """Service for interacting with the highlighting microservice."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the HighlightService."""
        self.base_url = base_url

    def highlight_key_quotes(
        self, query: str, assertions: list[AssertionModel]
    ) -> list[AssertionModel]:
        """
        Highlight the key quotes in the given assertion.

        Assumes that the given assertion model has only one citation. pre-process using .to_atomic_assertions if that is not the case.
        """
        for assertion in assertions:
            for citation in assertion.citations:
                citation.highlight = self._highlight_api(
                    query, assertion.assertion, citation.text
                )

        return assertions

    def _highlight_api(
        self, user_query: str, assertion: str, source_passage: str
    ) -> str:
        """
        Send a request to the highlighting microservice and return the highlighted sentences.

        Args:
            user_query (str): The user's query.
            assertion (str): The assertion to be checked.
            source_passage (str): The source passage to be analyzed.

        Returns:
            List[str]: A list of highlighted sentences.
        """
        url = f"{self.base_url}/highlight"

        payload = json.dumps(
            {
                "user_query": user_query,
                "assertion": assertion,
                "source_passages": [source_passage],
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}

        req = Request(url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))[
                    "highlighted_sentences"
                ][0]
        except URLError as e:
            raise Exception(f"Error communicating with highlighting service: {str(e)}")

        return ""
