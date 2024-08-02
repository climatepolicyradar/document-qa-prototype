import src.config as config
from src.online.retrieval import get_chroma_vector_store

import pytest

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma


@pytest.fixture
def vector_store() -> Chroma:
    """Get local vector store for testing."""
    encoder = HuggingFaceBgeEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Patch to force the test to use the local Chroma DB
    config.USE_LOCAL_CHROMA_DB = True
    vector_store = get_chroma_vector_store("cpr_documents_langchain", encoder)

    return vector_store
