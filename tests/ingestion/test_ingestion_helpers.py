import json
import pytest

from math import ceil
from cpr_data_access.models import BaseDocument

from src.ingestion.document_converter import CPRToLangChainDocumentConverter
from src.ingestion.text_splitter import CPRTextSplitter
from src.ingestion.utils import BLOCK_SEPARATOR


@pytest.fixture
def document():
    doc_path = "./data/documents/"
    return BaseDocument.load_from_local(
        doc_path, document_id="CCLW.executive.8737.1424"
    )


@pytest.mark.parametrize(
    ("n_text_blocks_in_chunk", "last_chunk_size", "num_chunks"),
    [(5, 1, 182), (10, 6, 91), (10000000, 906, 1)],
)
def test_conversion_and_splitting(
    document, n_text_blocks_in_chunk, last_chunk_size, num_chunks
):
    converter = CPRToLangChainDocumentConverter()
    splitter = CPRTextSplitter()

    langchain_doc = converter.convert(
        document, n_text_blocks_in_chunk=n_text_blocks_in_chunk
    )

    number_of_text_blocks = len(document.text_blocks)
    print(number_of_text_blocks)
    expected_number_of_chunks = ceil(number_of_text_blocks / n_text_blocks_in_chunk)

    assert expected_number_of_chunks == len(
        langchain_doc.page_content.split(BLOCK_SEPARATOR)
    )
    assert expected_number_of_chunks == len(
        json.loads(langchain_doc.metadata["chunk_metadata"])
    )

    chunks = splitter.split_documents([langchain_doc])

    assert len(chunks) == expected_number_of_chunks
    assert all([BLOCK_SEPARATOR not in chunk.page_content for chunk in chunks])
    assert all(
        [
            len(json.loads(chunk.metadata["text_block_ids"])) == n_text_blocks_in_chunk
            for chunk in chunks[:-1]
        ]
    )
    assert len(json.loads(chunks[-1].metadata["text_block_ids"])) == last_chunk_size
    assert len(chunks) == num_chunks
