"""Creates an on-disk vector store from a collection of documents."""
import uuid
from pathlib import Path

import typer
from cpr_data_access.models import Dataset, BaseDocument
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tqdm.auto import tqdm

from src.prompts.validation import text_is_too_long_for_model
from src.ingestion.document_converter import CPRToLangChainDocumentConverter
from src.ingestion.text_splitter import CPRTextSplitter
from src.ingestion.utils import ChromaBGEEmbeddingFunction
from src.online.retrieval import get_chroma_client
from src import config
from src.logger import get_logger

LOGGER = get_logger(__name__)


def run_ingestion_pipeline(
    doc_input_dir: Path = config.DOCUMENT_DIR,
):
    """
    Run ingestion pipeline.

    :param Path doc_input_dir: input dir containing pipeline opensearch_input JSONs.
        Defaults to Path("./data/documents")
    """

    _jsons_in_dir = list(doc_input_dir.glob("*.json"))
    LOGGER.info(
        f"Loading documents from {doc_input_dir} containing {len(_jsons_in_dir)} JSON files."
    )

    dataset = (
        Dataset(BaseDocument)
        .load_from_local(str(doc_input_dir))
        .filter_by_language("en")
    )

    document_converter = CPRToLangChainDocumentConverter()

    documents = [document_converter.convert(doc) for doc in dataset]

    text_splitter = CPRTextSplitter()
    LOGGER.info(f"Number of documents: {len(documents)}")
    chunked_documents = text_splitter.split_documents(documents)
    LOGGER.info(f"Number of chunked documents: {len(chunked_documents)}")

    chunk_too_long = [
        text_is_too_long_for_model(chunk.page_content, config.EMBEDDING_MODEL_NAME)
        for chunk in chunked_documents
    ]

    if any(chunk_too_long):
        LOGGER.warning(
            f"{sum(chunk_too_long)} chunks out of {len(chunk_too_long)} are too long for the model."
        )

    embed_model = HuggingFaceBgeEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    if config.USE_LOCAL_CHROMA_DB:
        assert not (
            config.CHROMA_DB_PATH / "chroma.sqlite3"
        ).exists(), "Chroma DB already exists, please delete old files."

    chroma_client = get_chroma_client()
    chroma_embedder = ChromaBGEEmbeddingFunction(config.EMBEDDING_MODEL_NAME)

    if config.USE_LOCAL_CHROMA_DB:
        langchain_db = Chroma.from_documents(
            chunked_documents,
            embed_model,
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

    else:
        typer.confirm(
            "This will reset the database. Are you sure you want to continue?",
            abort=True,
        )

        chroma_client.reset()  # resets the database
        collection = chroma_client.create_collection(
            config.CHROMA_COLLECTION_NAME, embedding_function=chroma_embedder
        )
        # TODO: we could probably batch ingestion
        for doc in tqdm(chunked_documents):
            collection.add(
                ids=str(uuid.uuid1()),
                metadatas=doc.metadata,
                documents=doc.page_content,
            )

        # tell LangChain to use our client and collection name
        langchain_db = Chroma(
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embed_model,
        )

    LOGGER.info(f"Number of documents: {len(documents)}")

    test_query_results = langchain_db.similarity_search("climate")
    if len(test_query_results) > 0:
        LOGGER.info("Success!")
    else:
        LOGGER.error("Test query failed. You might want to check the index.")


if __name__ == "__main__":
    typer.run(run_ingestion_pipeline)
