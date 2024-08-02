import copy
import json
from typing import List, Optional
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import TextSplitter

from src.ingestion.utils import BLOCK_SEPARATOR, transform_metadata


class CPRTextSplitter(TextSplitter):
    """
    Custom TextSplitter for CPR Documents

    This text-splitter uses the blocks defined by the `CPRToLangChainDocumentConverter` for the splitting,
    and hence the default arguments such as chunk_overlap or chunk_size are not used.
    """

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[LangChainDocument]:
        """
        Create documents from a list of texts.

        NOTE: this is a modified version of the TextSplitter method with the main difference being the
        loading and assignment of the chunk_metadata field in the metadata dictionary.
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            metadata = copy.deepcopy(_metadatas[i])
            chunk_metadata = json.loads(metadata.pop("chunk_metadata"))
            for split_idx, chunk in enumerate(self.split_text(text)):
                new_doc = LangChainDocument(
                    page_content=chunk,
                    metadata=transform_metadata(metadata)
                    | transform_metadata(chunk_metadata[split_idx]),
                )
                documents.append(new_doc)
        return documents

    def split_text(self, text: str) -> List[str]:
        """Splits the document text based on the block separator it was joined with."""
        return [t for t in text.split(BLOCK_SEPARATOR) if t != ""]
