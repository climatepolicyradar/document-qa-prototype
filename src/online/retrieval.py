"""Helper functions for retrieval."""
from pathlib import Path
import json
from typing import List, Optional, Iterable

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document as LangChainDocument

from src import config
from src.logger import get_logger

LOGGER = get_logger(__name__)


def get_chroma_client() -> ClientAPI:
    """
    Get a client used to interact with a local or server-based Chroma instance.

    Whether the client is local or server-based is determined by the USE_LOCAL_CHROMA_DB
    environment variable.
    """
    if config.USE_LOCAL_CHROMA_DB:
        return chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    else:
        return chromadb.HttpClient(
            host=config.CHROMA_SERVER_HOST or "localhost",
            settings=ChromaSettings(allow_reset=True),
        )


def get_chroma_vector_store(
    collection_name: str, embedding_function: Embeddings
) -> Chroma:
    """
    Get a Chroma vector store that can be used in Langchain.

    Whether the vector store is local or server-based is determined by the
    USE_LOCAL_CHROMA_DB environment variable.
    """

    db = get_chroma_client()

    return Chroma(
        client=db,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )


def get_available_document_ids_from_database(
    vector_store: Chroma, page_size: int = 1000
) -> Iterable[str]:
    """
    Get unique document IDs from the vector store.

    Works by getting all the documents where the next_chunk_id is "None", then
    extracting the document_id from each.
    """
    # TODO: this doesn't seem to be returning all the document IDs. Although, this
    # could also be an issue with the database or the ingest process.

    _number_returned = page_size
    idx = 0
    document_ids = list()

    while _number_returned == page_size:
        _retrieved_results = vector_store.get(
            include=["metadatas"],
            where={"next_chunk_id": "None"},
            limit=page_size,
            offset=idx * page_size,
        )
        _number_returned = len(_retrieved_results["metadatas"])
        for m in _retrieved_results["metadatas"]:
            document_ids.append(m["document_id"])

        idx += 1

    duplicate_ids = set([id for id in document_ids if document_ids.count(id) > 1])

    if len(duplicate_ids) > 0:
        LOGGER.warning(
            f"Duplicate document IDs found: {duplicate_ids}. This probably means that there's an issue with the index."
        )

    return set(document_ids)


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


def expand_window(
    vector_store: VectorStore,
    window_radius: int,
    docs: List[LangChainDocument],
) -> List[List[LangChainDocument]]:
    """
    Expands the window of documents to include the neighbouring documents.

    Args:
        vector_store: The VectorStoreRetriever to use for retrieval
        window_radius: The radius of the window to expand (i.e. how many neighbours to include in each direction)
        docs: The list of documents to expand the window around
    """
    LOGGER.info(f"Expanding window with radius {window_radius}")
    expanded_docs = []
    for doc in docs:
        window = [doc]
        for _ in range(window_radius):
            prev_doc, next_doc = get_neighbouring_docs(vector_store, window[0])

            if prev_doc is None and next_doc is None:
                LOGGER.warning(
                    f"No neighbouring documents found for {doc.metadata['document_id']}, chunk {doc.metadata['chunk_id']}. This is probably an error, unless the document only contains one chunk."
                )

            if prev_doc is not None:
                window.insert(0, prev_doc)
            if next_doc is not None:
                window.append(next_doc)
        expanded_docs.append(window)
    return expanded_docs


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
