import asyncio
import json
from anyio import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
from src.models.data_models import Prompt, RAGRequest, Scenario
from src.online.inference import LLMTypes
from src import config
from src.models.data_models import QAPair

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

    result = app_context["rag_controller"].run_rag_pipeline(
        query=request.query, scenario=request.as_scenario(dc)
    )

    if not ec.did_system_respond(result):
        LOGGER.info(
            f"🔍 System did not respond to the user query: {result.get_answer()}"
        )
        scenario = Scenario(
            prompt=Prompt.from_template("response/summarise_simple"),
            model="mistral-nemo",
            generation_engine=LLMTypes.VERTEX_AI.value,
        )
        summary = app_context["rag_controller"].run_llm(
            scenario, {"query_str": result.rag_response.retrieved_passages_as_string()}
        )
        LOGGER.info(f"🔍 System summarised the query: {result.get_answer()}")
        result.rag_response.text += f"\n\nWe found the following sources that may help answer the question:\n\n {summary}"

    db_save = QAPair.from_end_to_end_generation(result, "prototype")
    db_save.save()

    return_json = result.model_dump()
    return return_json


@app.get("/document/{document_id}")
def get_document_data(document_id: str) -> BaseDocument:
    """
    Get a document by ID.

    :param str document_id: The document ID
    :return BaseDocument: The document
    """
    return dc.create_base_document(document_id)


@app.get("/documents")
def get_documents():
    """Returns documents and their metadata available for the tool"""
    metadata_path = Path("data/document_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    """Returns a document and its metadata available for the tool"""
    metadata_path = Path("data/document_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return [d for d in metadata if d["id"] == document_id][0]


@app.websocket("/ws/stream_rag/")
async def stream_rag(websocket: WebSocket):
    """Stream RAG (Retrieval-Augmented Generation) on a document."""
    await websocket.accept()
    await websocket.send_json({"type": "ai", "content": "Connected to RAG stream"})
    input_data = await websocket.receive_json()
    request = RAGRequest(**input_data)

    try:
        # messy messy messy
        sc = ScenarioController.from_config(api_config)
        config_sc = sc.scenarios[
            0
        ]  # TODO - this is how we A/B Test! Pick scenarios from config
        request.prompt_template = config_sc.prompt.prompt_template
        request.model = config_sc.model
        request.generation_engine = LLMTypes(config_sc.generation_engine)

        current_sc = request.as_scenario(dc)

        result = await app_context["rag_controller"].stream_rag_pipeline(
            query=request.query, scenario=current_sc, websocket=websocket
        )

        await websocket.send_json({"type": "final", "result": result.model_dump()})
        # await websocket.send_json({'type': 'final', 'result': {'answer': 'test'}})
    except WebSocketDisconnect:
        await websocket.close()


@app.post("/highlights/{source_id}")
async def get_highlights(source_id: str):
    """Get highlights from a document."""
    qa_pair = QAPair.get_by_source_id(source_id)
    gen_model = qa_pair.to_end_to_end_generation()

    assert gen_model.rag_response is not None, "RAG response is None"

    assertions = app_context["rag_controller"].extract_assertions_from_answer(
        gen_model.get_answer()
    )

    LOGGER.info(gen_model.rag_request.query)
    LOGGER.info(gen_model.rag_request.document_id)

    async def process_assertion(assertion):
        return {
            "citations": assertion.citations,
            "answerSubstring": assertion.assertion,
            "uuid": assertion.uuid,
            "citationSubstring": await app_context[
                "rag_controller"
            ].highlight_key_quotes(
                gen_model.rag_request.query,
                assertion,
                gen_model.rag_response.retrieved_documents,  # type: ignore
            ),
        }

    LOGGER.info("🚀 Launching parallel highlight processing")
    all_assertions = [
        atomic
        for assertion in assertions
        for atomic in assertion.to_atomic_assertions()
    ]

    highlights = await asyncio.gather(
        *[process_assertion(assertion) for assertion in all_assertions]
    )

    LOGGER.info("🧠 Consolidating highlights by UUID")
    consolidated_highlights = {}
    for highlight in highlights:
        uuid = highlight["uuid"]
        if uuid not in consolidated_highlights:
            consolidated_highlights[uuid] = {
                "answerSubstring": highlight["answerSubstring"],
                "citationSubstrings": {},
            }
        consolidated_highlights[uuid]["citationSubstrings"][
            highlight["citations"][0]
        ] = highlight["citationSubstring"]

    highlights = list(consolidated_highlights.values())
    LOGGER.info(f"🎭 Consolidated {len(highlights)} unique assertions")
    LOGGER.info(f"✅ Processed {len(highlights)} highlights in parallel")

    return highlights


@app.post("/evaluate/{source_id}")
async def evaluate(source_id: str):
    """Evaluate an answer."""
    qa_pair = QAPair.get_by_source_id(source_id)
    gen_model = qa_pair.to_end_to_end_generation()

    evals = await ec.evaluate_async(gen_model)
    print(evals)
    return evals
