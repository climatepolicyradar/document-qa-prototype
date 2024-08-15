from enum import Enum
import json
import jinja2
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Optional
import datetime
from wandb.sdk.data_types.trace_tree import Trace
from cpr_data_access.models import BaseDocument
from langchain_core.prompts import ChatPromptTemplate
from peewee import (
    Model,
    AutoField,
    TextField,
    CharField,
    UUIDField,
    DateTimeField,
    ForeignKeyField,
)
from playhouse.postgres_ext import BinaryJSONField
from src.controllers.DocumentController import DocumentController
from src.flows.utils import get_db

import uuid
import hashlib

import src.config as config
from src.online.inference import LLMTypes

from src.prompts.template_building import (
    get_citation_template,
    jinja_template_loader,
    make_qa_prompt,
    system_prompt,
)
from src.config import root_templates_folder


try:
    db = get_db()
except Exception as e:
    print("Error connecting to database: ", e)
    db = None


class Prompt(BaseModel):
    """Represents a prompt template for generating responses."""

    prompt_template: str
    prompt_content: jinja2.Template

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_template(cls, prompt_template: str) -> "Prompt":
        """Returns a Prompt object from a prompt template."""
        return cls(
            prompt_template=prompt_template,
            prompt_content=jinja_template_loader(
                (root_templates_folder / f"{prompt_template}.txt")
            ),
        )

    def make_qa_prompt(self) -> ChatPromptTemplate:
        """Returns a full citation template with RAG generation policy for use in RAG pipeline"""
        return make_qa_prompt(
            user_prompt_template=get_citation_template(self.prompt_template),
            system_prompt=system_prompt,
        )


class Scenario(BaseModel):
    """
    Represents a single scenario for running the pipeline:

    On this document, with this model, and this prompt.
    """

    id: str = str(uuid.uuid4())
    model: str
    generation_engine: str
    prompt: Prompt
    document: Optional[BaseDocument] = None
    retrieval_window: Optional[int] = 1
    top_k_retrieval_results: Optional[int] = 10
    src_config: Optional[dict] = None

    def get_config(self) -> dict:
        """Returns the config dictionary"""
        if self.src_config is None:
            self.src_config = {}
        return self.src_config


class QueryType(Enum):
    """Enum for the type of query"""

    USER = "user"
    SYNTHETIC = "synthetic"
    TEST = "test"
    PRODUCT = "product"


class Query(BaseModel):
    """Query object"""

    text: str
    type: QueryType
    timestamp: datetime.datetime
    document_id: Optional[str] = None
    prompt_template: Optional[str] = None
    db_id: Optional[int] = None
    user: Optional[str] = None
    model: Optional[str] = None
    uuid: Optional[str] = None
    tag: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def validate_metadata_fields(cls, data: dict) -> dict:
        """Validates that the appropriate fields are present for the query type."""
        if data["type"] == QueryType.USER:
            assert "user" in data, "user must be provided for user queries"
        elif data["type"] == QueryType.SYNTHETIC:
            assert all(
                [field in data for field in ["model", "prompt_template"]]
            ), "model and prompt_template must be provided for synthetic queries"

        return data

    @model_validator(mode="before")
    @classmethod
    def set_uuid(cls, data: dict) -> dict:
        """Sets the UUID for the query."""
        if data.get("uuid") is None:
            _unique_id = f"{data['text']}_{data['type']}_{data['timestamp']}_{data.get('document_id')}_{data.get('user')}"
            data["uuid"] = hashlib.md5(_unique_id.encode()).hexdigest()
        return data

    @classmethod
    def from_response(
        cls,
        text: str,
        document_id: str,
        prompt_template: str,
        model: str,
        tag: Optional[str] = None,
    ) -> "Query":
        """Parses the json from an LLM generation into query."""
        return cls(
            text=text,
            type=QueryType.SYNTHETIC,
            timestamp=datetime.datetime.now(),
            model=model,
            prompt_template=prompt_template,
            document_id=document_id,
            tag=tag,
        )


class DBQuery(Model):
    """A database model for a query to the RAG pipeline"""

    id = AutoField()
    text = TextField()
    query_type = CharField(null=True)
    document_id = CharField(null=True)
    prompt = TextField(null=True)
    tag = CharField(null=True)
    user = CharField(null=True)
    model = CharField(null=True)
    uuid = UUIDField(null=True)
    metadata = BinaryJSONField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def from_query(cls, query: Query):
        """Converts a Query object to a DBQuery object."""
        return cls(
            document_id=query.document_id,
            model=query.model,
            prompt=query.prompt_template,
            text=query.text,
            query_type=query.type.value.lower().replace("querytype.", ""),
            created_at=query.timestamp,
            updated_at=datetime.datetime.now(),
            user=query.user,
            uuid=query.uuid,
            tag=query.tag,
        )

    def to_query(self) -> Query:
        """Converts a DBQuery object to a Query object."""
        return Query(
            text=str(self.text),
            type=QueryType(str(self.query_type).lower().replace("querytype.", "")),
            document_id=str(self.document_id),
            prompt_template=str(self.prompt),
            timestamp=self.created_at,  # pyright: ignore
            user=str(self.user),
            model=str(self.model),
            uuid=str(self.uuid),
            tag=str(self.tag),
            db_id=int(self.id),  # pyright: ignore
        )

    class Meta:
        """Set DB for the model"""

        database = db


class QAPair(Model):
    """Represents a Question-Answer pair in the database."""

    id = AutoField()
    document_id = CharField(null=True)
    model = CharField(null=True)
    prompt = TextField(null=True)
    pipeline_id = CharField(null=True)
    source_id = CharField(null=True)
    query_id = ForeignKeyField(DBQuery, null=True)
    question = TextField(null=True)
    answer = TextField(null=True)
    evals = BinaryJSONField(null=True)
    metadata = BinaryJSONField(null=True)
    status = CharField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)
    generation = BinaryJSONField(null=True)  # serialized EndToEndGeneration

    class Meta:
        """Set DB for the model"""

        database = db

    @classmethod
    def get_by_source_id(cls, source_id: str) -> "QAPair":
        """Returns a QAPair object by source_id."""
        return cls.select().where(cls.source_id == source_id).first()

    @classmethod
    def from_end_to_end_generation(cls, generation: "EndToEndGeneration", tag: str):
        """Converts an EndToEndGeneration object to a QAPair object."""
        return cls(
            document_id=generation.rag_request.document_id,
            model=generation.rag_request.model,
            prompt=generation.rag_request.prompt_template,
            pipeline_id=tag,
            question=generation.rag_request.query,
            answer=generation.get_answer(False),
            evals={},
            metadata={},
            source_id=generation.uuid,
            generation=generation.model_dump_json(),
        )

    def to_end_to_end_generation(self) -> "EndToEndGeneration":
        """Converts the QAPair object to an EndToEndGeneration object."""
        gen_dict = json.loads(self.generation)
        rag_request = RAGRequest(**gen_dict["rag_request"])
        rag_response = RAGResponse(**gen_dict["rag_response"])
        return EndToEndGeneration(
            config=gen_dict["config"],
            rag_request=rag_request,
            rag_response=rag_response,
            error=gen_dict["error"],
            uuid=gen_dict["uuid"],
        )


class RAGRequest(BaseModel):
    """Request object for the RAG pipeline."""

    model_config = ConfigDict(use_enum_values=True)

    query: str
    document_id: str
    top_k: int = 10
    generation_engine: LLMTypes = LLMTypes.OPENAI
    mock_generation: bool = False
    user: Optional[str] = ""
    model: Optional[str] = ""
    prompt_template: str = "FAITHFULQA_SCHIMANSKI_CITATION_QA_TEMPLATE_MODIFIED"
    retrieval_window: int = 1
    config: Optional[str] = "src/configs/answer_config.yaml"

    def as_scenario(self, dc: DocumentController) -> Scenario:
        """Returns the RAGRequest as a Scenario object."""
        return Scenario(
            model=self.model if self.model else "",
            generation_engine=str(self.generation_engine.value),
            prompt=Prompt.from_template(prompt_template=self.prompt_template),
            document=dc.create_base_document(document_id=self.document_id),
            retrieval_window=self.retrieval_window,
            top_k_retrieval_results=self.top_k,
        )

    @classmethod
    def from_scenario(cls, query: str, scenario: Scenario) -> "RAGRequest":
        """Returns the Scenario object as a RAGRequest object."""
        return cls(
            query=query,
            model=scenario.model,
            generation_engine=LLMTypes(scenario.generation_engine),
            prompt_template=scenario.prompt.prompt_template,
            document_id=scenario.document.document_id if scenario.document else "",
            retrieval_window=scenario.retrieval_window
            if scenario.retrieval_window
            else 1,
            top_k=scenario.top_k_retrieval_results or 5,
        )


class RAGResponse(BaseModel):
    """Response object for the RAG pipeline"""

    text: str
    retrieved_documents: list[dict]
    query: str
    highlights: Optional[list[str]] = None

    # LangChain uses pydantic v1 internally, so can't pass LangChainDocuments here

    @property
    def citation_numbers(self) -> set[int]:
        """Returns the citation numbers that are used to code the sources"""
        return set(range(0, len(self.retrieved_documents)))

    def refused_answer(self) -> bool:
        """
        Whether the model refused to answer the question.

        Use heuristics on the generated text to determine this.
        """

        refusal_phrases = ["do not provide information", "cannot provide an answer"]

        if any(phrase in self.text.lower() for phrase in refusal_phrases):
            return True

        return False

    def extract_inner_monologue(self) -> dict:
        """Extract the inner monologue from the RAG answer. Inner monologue is the text between #COT# and #/COT#"""
        result = {
            "inner_monologue": "",
            "answer": "",
        }
        if "#COT#" in self.text and "#/COT#" in self.text:
            result["inner_monologue"] = self.text.split("#COT#")[1].split("#/COT#")[0]
            result["answer"] = self.text.split("#/COT#")[1]
        else:
            result["answer"] = self.text

        return result

    def retrieved_passages_as_string(self) -> str:
        """Returns a string representation of the retrieved passages."""

        return "\n".join(
            [
                f"[{idx+1}]: {p['page_content']}"
                for idx, p in enumerate(self.retrieved_documents)
            ]
        )

    def retrieved_windows_as_string(self) -> str:
        """Returns a string representation of windows, which are lists of retrieved passages."""

        within_window_separator: str = "\n"
        window_texts: list[str] = [
            within_window_separator.join(i["metadata"]["text_block_window"] for i in w)
            for w in self.retrieved_documents
        ]

        return "\n\n".join(
            f"**[{idx}]**\n {window_text}"
            for idx, window_text in enumerate(window_texts)
        )


class AssertionModel(BaseModel):
    """Model for assertions extracted from RAG responses."""

    assertion: str
    citations: list[str]

    @model_validator(mode="before")
    @classmethod
    def validate_assertion(cls, data: dict) -> dict:
        """Validates the assertion and citations."""
        assert "assertion" in data, "Assertion must be provided"
        assert isinstance(data["assertion"], str), "Assertion must be a string"
        assert "citations" in data, "Citations must be provided"
        assert isinstance(
            data["citations"], list
        ), "Citations must be a list of strings"
        return data

    def citations_as_string(self) -> str:
        """Returns a string representation of the citations."""
        return ",\n".join(self.citations)

    def __str__(self) -> str:
        """Returns a string representation of the AssertionModel object."""
        return f"AssertionModel(assertion={self.assertion}, citations={self.citations})"


class EndToEndGeneration(BaseModel):
    """
    Generation with config, a RAG response, and potentially an error.

    TODO: The rag_request property should really be a query:str and a scenario as these are the domain model abstractions and RagRequest is a transport layer abstraction. But it's used in a lot of places so would take a bit too long to refactor right now. When we do refactor, we should also update the consumer code to use getter methods to ensure that refactor doesn't need to happen again
    """

    config: dict
    rag_request: RAGRequest
    rag_response: Optional[RAGResponse] = None
    error: Optional[str] = None
    uuid: Optional[str] = None

    def __str__(self) -> str:
        """Returns a string representation of the EndToEndGeneration object."""
        return (
            f"EndToEndGeneration({self.rag_request.query}, {self.rag_response.__str__})"
        )

    def get_answer(self, remove_cot: bool = True) -> str:
        """Returns the answer from the RAG response. If remove_cot is True, the inner monologue is removed before returning, otherwise the full response is returned."""
        if self.rag_response is None:
            return ""

        if remove_cot:
            return self.rag_response.extract_inner_monologue()["answer"]

        return self.rag_response.text

    @model_validator(mode="before")
    @classmethod
    def set_uuid(cls, data: dict) -> dict:
        """Sets the UUID for the query."""
        query = data["rag_request"].query
        response = data["rag_response"].text if data.get("rag_response") else "None"
        document_id = data["rag_request"].document_id

        if data.get("uuid") is None:
            _unique_id = "_".join([query, response, document_id])
            data["uuid"] = hashlib.md5(_unique_id.encode()).hexdigest()
        return data

    def to_db_model(self, tag: str, query_id: Optional[str] = None) -> QAPair:
        """Converts the EndToEndGeneration object to a QAPair object."""
        return QAPair(
            document_id=self.rag_request.document_id,
            model=self.rag_request.model,
            prompt=self.rag_request.prompt_template,
            pipeline_id=tag,
            question=self.rag_request.query,
            query_id=query_id,
            answer=self.get_answer(False),
            evals={},
            metadata={},
            generation=json.dumps(self.model_dump()),
        )


class RAGLog(BaseModel):
    """Log object for the RAG pipeline."""

    id: uuid.UUID = uuid.uuid4()
    user: Optional[str]
    query: str
    document_id: str
    top_k: int
    retrieved_documents: list[dict]
    start_time: datetime.datetime
    end_time: datetime.datetime
    generation: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_request_and_response(
        cls,
        rag_response: RAGResponse,
        rag_request: RAGRequest,
        start_time: datetime.datetime,
    ) -> "RAGLog":
        """Creates the log trace from the request and response."""
        return cls(
            user=rag_request.user,
            query=rag_request.query,
            document_id=rag_request.document_id,
            top_k=rag_request.top_k,
            retrieved_documents=rag_response.retrieved_documents,
            start_time=start_time,
            end_time=datetime.datetime.now(),
            generation=rag_response.text,
        )

    def log_to_wandb(self):
        """Log to weights and biases. Requires a wandb run to have been created."""

        if config.WANDB_ENABLED:
            wandb_trace = Trace(
                name="RAGlog",
                status_code="success",  # TODO: log failures
                metadata={
                    "id": self.id,
                    "user": self.user,
                    "document_id": self.document_id,
                },
                start_time_ms=int(self.start_time.timestamp() * 1000),
                end_time_ms=int(self.end_time.timestamp() * 1000),
                inputs={
                    "query": self.query,
                    "top_k": self.top_k,
                },
                outputs={
                    "retrieved_documents": self.retrieved_documents,
                    "generation": self.generation,
                },
            )

            wandb_trace.log(f"{self.id}")
