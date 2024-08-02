import json
from typing import cast

from chromadb.api.types import EmbeddingFunction, Embeddings
from chromadb import Documents
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


BLOCK_SEPARATOR = "\n\n----------------\n\n"


def transform_metadata(metadata: dict) -> dict:
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = json.dumps(value)
        elif not isinstance(value, (str, int, float, bool)):
            metadata[key] = str(value)
    return metadata


class ChromaBGEEmbeddingFunction(EmbeddingFunction):
    """
    A typing fix for the langchain embedder to make it compatible with Chroma.

    TODO: this seems to slow down the embedding process
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = HuggingFaceBgeEmbeddings(model_name=self.model_name)

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of Documents (strings)."""
        return cast(Embeddings, self.embedder.embed_documents(input))
