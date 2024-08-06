import wandb

from typing import Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.globals import set_verbose, set_debug
from dotenv import load_dotenv, find_dotenv
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import Scenario, ScenarioController

from src.logger import get_logger
from src.models.data_models import RAGRequest
from src import config


LOGGER = get_logger(__name__)
DEBUG = False


if DEBUG:
    set_debug(True)
else:
    set_verbose(True)


app_context = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application lifespan.

    Commands to run on startup go before the yield, and commands to run on shutdown go
    after the yield.
    """
    LOGGER.info("Loading environment variables...")
    load_dotenv(find_dotenv())
    LOGGER.info("Environment variables loaded...", extra={".env found?": find_dotenv()})

    # Enable logging to weights and biases prompts unless explicitly disabled
    # by setting the DISABLE_WANDB environment variable to "true" or "1"
    if config.WANDB_ENABLED:
        LOGGER.info("Enabling weights and biases logging...")
        wandb.init(project=config.WANDB_PROJECT_NAME)

    LOGGER.info("Getting vector store index...")
    app_context["encoder"] = HuggingFaceBgeEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    app_context["rag_controller"] = RagController()
    LOGGER.info("Vector store index ready...")

    yield


LOGGER.info("Creating FastAPI app...")
app = FastAPI(lifespan=lifespan)
LOGGER.info("App and context created")


@app.get("/health")
async def get_health():
    """
    Get application health.

    TODO: fill in implementation with real health check.
    """
    LOGGER.info("Health check requested...")
    return {"status": "OK"}

@app.get("/rag/{document_id}")
def do_rag(request: RAGRequest) -> dict:
    """
    Perform RAG (Retrieval-Augmented Generation) on a document.
    :param str query: The text query
    :param str document_id: The document ID
    :param str config: The name of the config file to use, which will be loaded from the config folder and must correspond to a config file name in the format of <config_name>.yaml
    :return dict: RAG result
    """
    assert request.config is not None, "Config name is required"
    assert request.query is not None, "Query is required"
    assert request.document_id is not None, "Document ID is required"

    sc = ScenarioController.from_config(config_name=request.config)

    return app_context['rag_controller'].run_rag_pipeline(
        query=request.query,
        scenario=request.as_scenario()
    ).model_dump()


@app.get("/document_ids")
def get_document_ids():
    """
    Get unique document IDs from the vector store.

    :return list: list of document IDs
    """
    
    ## TODO I feel maybe this array mangling should be in get_available_documents but need to look at the JSON deeper to see if there's other information we may want to not lose
    return {"document_ids": [leaf['value'] for leaf in app_context['rag_controller'].get_available_documents()['root']['children']['children']['children']]}
