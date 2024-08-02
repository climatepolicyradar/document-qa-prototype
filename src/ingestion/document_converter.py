from cpr_data_access.models import BaseDocument
from langchain_core.documents import Document as LangChainDocument

from src.ingestion.utils import BLOCK_SEPARATOR, transform_metadata


class CPRToLangChainDocumentConverter:
    """
    Convert CPR documents to Langchain documents.

    Joins the text of each block with a fixed separator, which can then be reverse-engineered
    for chunking in the TextSplitter.
    """

    INTRA_CHUNK_SEPARATOR = "\n"

    def __init__(self, embed_metadata: bool = False) -> None:
        self._embed_metadata = embed_metadata

    def convert(
        self, document: BaseDocument, n_text_blocks_in_chunk: int = 1
    ) -> LangChainDocument:
        """
        Convert a CPR document to a langchain document.

        Text blocks are separated by a fixed separator.

        Args:
            document (BaseDocument): The CPR document to convert.
            n_text_blocks_in_chunk (int): The number of text blocks to include in each chunk.
        """
        chunks, chunk_metadata = self._create_blocks_and_metadata(
            document, n_text_blocks_in_chunk
        )
        metadata = self._create_and_transform_metadata(document, chunk_metadata)

        return LangChainDocument(
            page_content=BLOCK_SEPARATOR.join(chunks),
            metadata=metadata,
        )

    def _create_blocks_and_metadata(
        self, document: BaseDocument, n_text_blocks_in_chunk
    ) -> tuple[list[str], list[dict]]:
        """
        Creates chunks and chunk-level metadata from the text blocks of a document.

        Uses the n_text_blocks_in_chunk parameter as a non-overlapping window size to merge text-blocks into chunks.
        """
        blocks = document.text_blocks or []
        chunks, chunk_metadata = [], []

        for i in range(0, len(blocks), n_text_blocks_in_chunk):
            _chunk_blocks = blocks[i : i + n_text_blocks_in_chunk]
            chunks.append(
                self.INTRA_CHUNK_SEPARATOR.join(
                    block.to_string() for block in _chunk_blocks
                )
            )
            text_block_ids = [block.text_block_id for block in _chunk_blocks]
            chunk_id = f"{document.document_id}_{'-'.join(text_block_ids)}"

            if chunk_metadata:
                chunk_metadata[-1]["next_chunk_id"] = chunk_id

            chunk_metadata.append(
                {
                    "text_block_ids": text_block_ids,
                    "chunk_id": chunk_id,
                    "next_chunk_id": None,
                }
            )

        return chunks, chunk_metadata

    @staticmethod
    def _create_and_transform_metadata(
        document: BaseDocument, chunk_metadata: list[dict]
    ) -> dict:
        """
        Creates the metadata object for the LangChain document.

        The metadata contains page-level metadata as well as the chunk-level metadata. The latter is
        a list of dictionaries with the same keys.

        The metadata dict is transformed to a format that can be stored in ChromaDB. The chunk_metadata
        field is converted to a JSON string, which needs conversion in the TextSplitter.
        """
        metadata = (
            document.model_dump(
                exclude={"text_blocks", "document_metadata", "page_metadata"}
            )
            | document.document_metadata.model_dump()
            | {"chunk_metadata": chunk_metadata}
        )

        metadata = transform_metadata(metadata)

        return metadata
