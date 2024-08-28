from datetime import datetime
import random
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain.globals import set_verbose, set_debug
from dotenv import load_dotenv, find_dotenv
from src.controllers.NotebookController import NotebookController
from src.controllers.DocumentController import DocumentController
from src.controllers.EvaluationController import EvaluationController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.logger import get_logger
from src.models.data_models import EndToEndGeneration, RAGRequest
from src.online.inference import LLMTypes
from src import config
from src.models.data_models import QAPair
from src.services.HighlightService import HighlightService
from src.controllers.LibraryManager import LibraryManager

config.logger.info("Here we go yo")  # I just want ruff to stop removing my import.

LOGGER = get_logger(__name__)
DEBUG = False


if DEBUG:
    set_debug(True)
else:
    set_verbose(True)

dc = DocumentController()
ec = EvaluationController()
api_config = "src/configs/answer_config.yaml"
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

notebook_controller = NotebookController()


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

    LOGGER.info(
        f"Running RAG pipeline with model {request.model} and prompt {request.prompt_template}"
    )

    result = app_context["rag_controller"].run_rag_pipeline(
        query=request.query,
        scenario=request.as_scenario(dc),
        return_early_on_guardrail_failure=True,
    )

    if result.rag_response.metadata.get("guardrails_failed"):
        return result.model_dump()

    # Only try to summarise retrieved passages ('no answer flow') if there are
    # passages to summarise and the answer was refused
    if (
        result.rag_response.refused_answer()
        and len(result.rag_response.retrieved_documents) > 0
    ):
        result = app_context["rag_controller"].execute_no_answer_flow(result)

    result.rag_response.augment_passages_with_metadata(request.document_id)
    try:
        # Save the answer to the database
        db_save = QAPair.from_end_to_end_generation(result, "prototype")
        db_save.save()

        # Create or update the notebook for this answer
        notebook = notebook_controller.update_notebook(request.notebook_uuid, db_save)
        result.add_metadata("notebook_uuid", notebook.uuid)
    except Exception as e:
        LOGGER.error(f"Error saving to database: {e}")
        raise e

    return_json = result.model_dump()
    return return_json


class NotebookResponse(BaseModel):
    """Holds the notebook and its answers."""

    id: int
    uuid: str
    name: str
    created_at: datetime
    updated_at: datetime
    is_shared: bool
    answers: list[EndToEndGeneration]


@app.get("/notebook/{notebook_uuid}")
def get_notebook(notebook_uuid: str) -> NotebookResponse:
    """Get a notebook by UUID."""
    result = notebook_controller.get_notebook_with_answers(notebook_uuid)

    # Pyright and peewee don't get along
    return NotebookResponse(
        id=result.id,  # type: ignore
        uuid=notebook_uuid,
        name=result.name,  # type: ignore
        created_at=result.created_at,  # type: ignore
        updated_at=result.updated_at,  # type: ignore
        is_shared=result.is_shared,  # type: ignore
        answers=[a.to_end_to_end_generation() for a in result.answers],  # type: ignore
    )


@app.get("/document/{document_id}")
def get_document_data(document_id: str) -> dict:
    """
    Get a document by ID.

    :param str document_id: The document ID
    :return BaseDocument: The document
    """
    return dc.get_metadata(document_id)


@app.get("/documents")
def get_documents():
    """Returns documents and their metadata available for the tool"""
    return LibraryManager().get_documents()


@app.get("/random")
def get_random_document():
    all_docs = LibraryManager().get_documents()
    if "rows" in all_docs:
        all_docs = all_docs["rows"]  # type: ignore

    if len(all_docs) == 0:
        raise HTTPException(status_code=404, detail="No documents found")

    random_doc = random.choice(all_docs)
    return random_doc


@app.post("/highlights/{source_id}")
async def get_highlights(source_id: str):
    """Get highlights from a document."""
    qa_pair = QAPair.get_by_source_id(source_id)
    gen_model = qa_pair.to_end_to_end_generation()

    assert gen_model.rag_response is not None, "RAG response is None"

    hs = HighlightService()

    return hs.highlight_key_quotes(
        gen_model.rag_response.query,
        gen_model.processed_generation_data.assertions,  # type: ignore
    )  # type: ignore


@app.post("/evaluate-all/{source_id}")
async def evaluate(source_id: str):
    """Evaluate an answer."""
    qa_pair = QAPair.get_by_source_id(source_id)
    gen_model = qa_pair.to_end_to_end_generation()

    evals = await ec.evaluate_async(gen_model)
    return evals


@app.post("/evaluate/{eval_id}/{source_id}")
def evaluate_single(eval_id: str, source_id: str):
    """Evaluate a single answer. Optimising for parallel calls from the frontend."""
    qa_pair = QAPair.get_by_source_id(source_id)
    gen_model = qa_pair.to_end_to_end_generation()

    mini_ec = EvaluationController()
    mini_ec.set_evaluators([eval_id])

    evals = mini_ec.evaluate(gen_model, eval_id)
    return evals
