from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain.globals import set_verbose, set_debug
from dotenv import load_dotenv, find_dotenv
from src.controllers.DocumentController import DocumentController
from src.controllers.EvaluationController import EvaluationController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from cpr_data_access.models import BaseDocument

from src.logger import get_logger
from src.models.data_models import RAGRequest
from src.online.inference import LLMTypes

LOGGER = get_logger(__name__)
DEBUG = False


if DEBUG:
    set_debug(True)
else:
    set_verbose(True)

dc = DocumentController()
ec = EvaluationController()
app_context = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application lifespan.

    Commands to run on startup go before the yield, and commands to run on shutdown go
    after the yield.
    """
    LOGGER.info("Loading environment variables...")
    LOGGER.info(f"Loading from {find_dotenv()}")
    load_dotenv(find_dotenv(), override=True)
    LOGGER.info("Environment variables loaded...", extra={".env found?": find_dotenv()})

    LOGGER.info("Getting vector store index...")
    app_context["rag_controller"] = RagController()
    LOGGER.info("Vector store index ready...")

    yield


LOGGER.info("Creating FastAPI app...")
app = FastAPI(lifespan=lifespan)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
LOGGER.info("App and context created")


@app.get("/")
async def get_health():
    """
    Get application health.

    TODO: fill in implementation with real health check.
    """
    LOGGER.info("Health check requested...")
    return {"status": "OK"}


@app.post("/rag")
def do_rag(request: RAGRequest) -> dict:
    """
    Perform RAG (Retrieval-Augmented Generation) on a document.

    :param str query: The text query
    :param str document_id: The document ID
    :return dict: RAG result
    """
    api_config = "src/configs/answer_config.yaml"

    assert request.query is not None, "Query is required"
    assert request.document_id is not None, "Document ID is required"

    sc = ScenarioController.from_config(api_config)

    # This is messy. Needs refactoring. Result of leaky abstraction that Matyas identified
    config_sc = sc.scenarios[
        0
    ]  # TODO - this is how we can A/B Test! Pick scenarios from config
    request.prompt_template = config_sc.prompt.prompt_template
    request.model = config_sc.model
    request.generation_engine = LLMTypes(config_sc.generation_engine)

    return (
        app_context["rag_controller"]
        .run_rag_pipeline(query=request.query, scenario=request.as_scenario(dc))
        .model_dump()
    )


@app.get("/document/{document_id}")
def get_document(document_id: str) -> BaseDocument:
    """
    Get a document by ID.

    :param str document_id: The document ID
    :return BaseDocument: The document
    """
    return dc.create_base_document(document_id)


@app.get("/document_ids")
def get_document_ids():
    """
    Get unique document IDs from the vector store.

    This function retrieves and returns a list of all unique document IDs
    available in the vector store for RAG processing.

    :return list: list of document IDs
    """

    ## TODO I feel maybe this array mangling should be in get_available_documents but need to look at the JSON deeper to see if there's other information we may want to not lose
    return {
        "document_ids": [
            leaf["value"]
            for leaf in app_context["rag_controller"].get_available_documents()["root"][
                "children"
            ]["children"]["children"]
        ]
    }
