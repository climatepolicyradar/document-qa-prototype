"""Helper functions for retrieval."""
from pathlib import Path
import json
from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document as LangChainDocument

from src.logger import get_logger

LOGGER = get_logger(__name__)


def get_available_documents_from_dir(documents_dir: Path) -> list[dict]:
    """
    Get the available documents.

    :return list: list of dicts describing the available documents with keys
        `document_id` and `document_metadata`
    """
    doc_info = []

    for path in documents_dir.glob("*.json"):
        doc = json.loads(path.read_text())

        doc_info.append(
            {
                "document_id": doc["document_id"],
                "document_metadata": doc["document_metadata"],
            }
        )

    return doc_info


def get_neighbouring_docs(
    vector_store: VectorStore,
    doc: LangChainDocument,
) -> tuple[Optional[LangChainDocument], Optional[LangChainDocument]]:
    """
    Retrieves a neighbouring document of the given document.

    Args:
        vector_store: The VectorStoreRetriever to use for retrieval
        doc: The document to get the neighbour of
    """

    _next_where_clause = {
        "$and": [
            {"chunk_id": doc.metadata["next_chunk_id"]},
            {"document_id": doc.metadata["document_id"]},
        ]
    }

    _previous_where_clause = {
        "$and": [
            {"next_chunk_id": doc.metadata["chunk_id"]},
            {"document_id": doc.metadata["document_id"]},
        ]
    }

    query_where_clause = {"$or": [_next_where_clause, _previous_where_clause]}

    _retrieved_results = vector_store.get(where=query_where_clause)  #  type: ignore

    if len(_retrieved_results["documents"]) == 0:
        return None, None

    prev_doc = None
    next_doc = None

    for idx, retrieved_doc in enumerate(_retrieved_results["metadatas"]):
        if retrieved_doc["chunk_id"] == doc.metadata["next_chunk_id"]:
            next_doc = LangChainDocument(
                page_content=_retrieved_results["documents"][idx],
                metadata=_retrieved_results["metadatas"][idx],
            )
        if retrieved_doc["next_chunk_id"] == doc.metadata["chunk_id"]:
            next_doc = LangChainDocument(
                page_content=_retrieved_results["documents"][idx],
                metadata=_retrieved_results["metadatas"][idx],
            )

    return prev_doc, next_doc
