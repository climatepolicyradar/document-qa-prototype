import json
import requests
from src.flows.utils import get_secret
from src.logger import get_logger

logger = get_logger(__name__)


class LibraryManager:
    """Manager for interacting with the RAG Library microservice."""

    def __init__(
        self, base_url: str = "https://rag-library.labs.climatepolicyradar.org"
    ):
        self.base_url = base_url
        self.api_token = get_secret("RAG_LIBRARY_API_TOKEN")

    def _get_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}"}

    def get_documents(self) -> list[dict]:
        """Get all documents from the library."""
        url = f"{self.base_url}/full_text/-/query.json"
        params = {
            "sql": "select document_id, name, description, slug, family_slug, publication_ts, geography, category, type, source, keyword from documents",
        }
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()

    def get_document_metadata(self, document_id: str) -> dict:
        """Get metadata for a document."""
        url = f"{self.base_url}/full_text/-/query.json"
        params = {
            "sql": "select * from documents where document_id = :document_id",
            "document_id": document_id,
        }
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        if len(response.json()["rows"]) == 0:
            return {}

        json_result = response.json()["rows"][0]

        # Pre JSON load the "languages", "topic", "hazard", "sector", "keyword", "framework", "instrument" fields
        for field in [
            "languages",
            "topic",
            "hazard",
            "sector",
            "keyword",
            "framework",
            "instrument",
        ]:
            try:
                json_result[field] = json.loads(json_result[field])
            except Exception:
                logger.error(f"Error loading {field} for document {document_id}")
                pass

        return json_result

    def get_text_blocks_around(
        self, document_id: str, text_block_id: int, N: int
    ) -> dict:
        """Get N text blocks before and after a given block for a document."""
        url = f"{self.base_url}/full_text/-/query.json"
        params = {
            "sql": "SELECT distinct(tb.text_block_id), tb.text, tb.page_number FROM text_blocks tb WHERE tb.document_id = :document_id AND CAST(tb.text_block_id AS INTEGER) BETWEEN (:text_block_id - :N) AND (:text_block_id + :N) ORDER BY tb.text_block_id",
            "document_id": document_id,
            "text_block_id": text_block_id,
            "N": N,
        }
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()

    def get_full_text(self, document_id: str) -> dict:
        """Get the full text of a document."""
        url = f"{self.base_url}/full_text/-/query.json"
        params = {
            "sql": "SELECT text_block_id, text, page_number FROM text_blocks WHERE document_id = :document_id ORDER BY text_block_id",
            "document_id": document_id,
        }
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()

    def get_section_containing_block(
        self, document_id: str, text_block_id: int
    ) -> dict:
        """Get the section of a document that contains a given text block."""
        url = f"{self.base_url}/full_text/-/query.json"
        params = {
            "sql": "WITH current_block AS (SELECT CAST(text_block_id AS INTEGER) AS current_block_number FROM text_blocks WHERE document_id = :document_id AND text_block_id = :text_block_id) SELECT tb.text_block_id, tb.text, tb.type, tb.type_confidence, tb.page_number FROM text_blocks tb, current_block cb WHERE tb.document_id = :document_id AND CAST(tb.text_block_id AS INTEGER) < cb.current_block_number AND tb.type IN ('sectionHeading', 'title') AND tb.type_confidence > 0.8 ORDER BY CAST(tb.text_block_id AS INTEGER) DESC LIMIT 1",
            "document_id": document_id,
            "text_block_id": text_block_id,
        }
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()
