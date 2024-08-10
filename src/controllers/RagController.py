from datetime import datetime
import json
import random
from src.controllers.DocumentController import DocumentController
from src.controllers.ScenarioController import Scenario
from cpr_data_access.models import BaseDocument
from typing import Optional
from langchain_core.messages.base import BaseMessage
from langchain_core.language_models.base import BaseLanguageModel
from src.controllers.VespaController import VespaController
from src.controllers.ObservabilityManager import ObservabilityManager

from src.logger import get_logger
from src.models.data_models import (
    AssertionModel,
    EndToEndGeneration,
    Prompt,
    Query,
    RAGRequest,
    RAGResponse,
)
from src.online.inference import get_llm
from src.dataset_creation.query_utils import (
    render_document_text_for_llm,
    sanitise_response_text,
)

from src.online.pipeline import rag_chain
from src import config


LOGGER = get_logger(__name__)


class RagController:
    """Controller for RAG operations"""

    def __init__(self, observe: bool = True):
        self.vespa = VespaController()

        self.observability = ObservabilityManager()
        self.observe = observe

    def get_llm(
        self, type: str, model: str, unfiltered: bool = False
    ) -> BaseLanguageModel:  # type: ignore
        """
        Returns an instantiated LLM of the given type and model code.

        TODO: Move the get_llm underlying function to here for nicer encapsulation
        """
        return get_llm(type, model, unfiltered)

    def get_llm_parameters(self, type: str, model: str):
        """Get the parameters for the given model. Only supported for vertex models right now"""
        if type == "vertexai":
            return config.VERTEX_MODEL_ENDPOINTS[model]["params"]
        else:
            return {}

    def get_available_documents(self) -> list[BaseDocument]:
        """
        Passthrough method to Vespa controller.

        Get the available documents in the Vespa database.
        """
        document_data = self.vespa.get_available_documents()
        dc = DocumentController()
        docs = [
            dc.create_base_document(doc["value"])
            for doc in document_data["root"]["children"][0]["children"][0]["children"]
        ]
        return docs

    def get_document_text(self, document_id: str):
        """
        Passthrough method to Vespa controller.

        Get the text of a document from the Vespa database.
        """
        return self.vespa.get_document_text(document_id)

    def run_llm(self, scenario: Scenario, prompt_data: dict) -> str:
        """Run an LLM with the given scenario, prompt key, and prompt data."""
        llm = self.get_llm(scenario.generation_engine, scenario.model)
        prompt = scenario.prompt.prompt_content.render(prompt_data)

        LOGGER.info(f"🤔 Running {scenario.model} and prompt: {prompt}")

        if self.observe:
            result = llm.invoke(
                prompt,
                config={"callbacks": [self.observability.get_tracing_callback()]},
                **self.get_llm_parameters(scenario.generation_engine, scenario.model),
            )
        else:
            result = llm.invoke(
                prompt,
                **self.get_llm_parameters(scenario.generation_engine, scenario.model),
            )

        LOGGER.info(f"Model: {scenario.model} -> {result}")
        return_str = ""
        if isinstance(result, BaseMessage):
            return_str = str(result.content)
        elif isinstance(result, str):
            return_str = result

        return return_str

    def generate_queries(
        self,
        scenario: Scenario,
        document: BaseDocument,
        seed_queries: list[Query],
        tag: Optional[str] = "",
    ) -> list[Query]:
        """
        Generate a set of queries using the given

        prompt template, model, document text and seed queries.
        """
        LOGGER.info(
            f"🔍 Generating queries for document: {document.document_id} {scenario}"
        )
        document_text = render_document_text_for_llm(document.text, scenario.model)

        llm = self.get_llm(scenario.generation_engine, scenario.model, unfiltered=True)

        prompt = scenario.prompt.prompt_content.render(
            seed_queries=random.sample(seed_queries, min(10, len(seed_queries))),
            document=document_text,
        )

        if self.observe:
            response = llm.invoke(
                prompt,
                config={"callbacks": [self.observability.get_tracing_callback()]},
                **self.get_llm_parameters(scenario.generation_engine, scenario.model),
            )
        else:
            response = llm.invoke(prompt)

        content = response.content if isinstance(response, BaseMessage) else response

        assert isinstance(content, str)
        LOGGER.info(f"💡 Queries: {content}")

        return self._parse_response_into_queries(
            content,
            document.document_id,
            scenario,
            tag,  # type: ignore
        )

    def run_rag_pipeline(
        self,
        query: str,
        scenario: Scenario,
    ) -> EndToEndGeneration:
        """Run the RAG pipeline for a given query and scenario."""

        LOGGER.info(f"***RUNNING RAG - query = `{query}`***")

        assert scenario.document is not None, "Scenario must have a document"

        start_time = datetime.now()

        llm = self.get_llm(scenario.generation_engine, scenario.model)

        LOGGER.info(f"🤔 Running document: {scenario.document.document_id}")

        rag_chain_with_source = rag_chain(
            llm=llm,
            retriever=self.vespa.retriever(scenario.document.document_id),  # type: ignore
            citation_template=scenario.prompt.make_qa_prompt(),
            scenario=scenario,
        )

        response = rag_chain_with_source.invoke(
            {
                "query_str": query,
                "document_id": scenario.document.document_id,
                "document_metadata_context_str": f"'{scenario.document.document_name}' pub. {scenario.document.document_metadata.publication_ts} (country:{scenario.document.document_metadata.geography})",
            },
            config={"callbacks": [self.observability.get_tracing_callback()]},
        )

        response_text = response["answer"]
        LOGGER.info(f"Response: {response_text}")

        end_time = datetime.now()
        duration = end_time - start_time
        LOGGER.info(f"Duration: {duration}")

        response = EndToEndGeneration(
            config=scenario.get_config(),
            rag_request=RAGRequest.from_scenario(query, scenario),
            rag_response=RAGResponse(
                text=response_text,
                retrieved_documents=[d.dict() for d in response["documents"]],
                query=query,
            ),
        )

        return response

    def extract_assertions_from_answer(
        self,
        answer: str,
    ) -> list[AssertionModel]:
        """Extract the assertions from the RAG answer."""
        assertions = self._extract_assertions(answer)
        return assertions

    def highlight_key_quotes(
        self,
        scenario: Scenario,
        query: str,
        assertion: AssertionModel,
        context_str: str,
    ) -> str:
        """Highlight the key quotes in the given assertion."""
        highlight_prompt_key = "response/extract_key_quotes"

        highlight_scenario = Scenario(
            model=scenario.model,
            generation_engine=scenario.generation_engine,
            prompt=Prompt.from_template(highlight_prompt_key),
        )
        args = {
            "query_str": query,
            "assertion": assertion.assertion,
            "cited_sources": assertion.citations_as_string(),
            "context_str": context_str,
        }

        LOGGER.info(f"🔍 Running highlight scenario with args: {args}")
        highlight_text = self.run_llm(highlight_scenario, args)
        LOGGER.info(f"🔍 Highlighted text: {highlight_text}")
        return highlight_text

    def _extract_assertions(self, rag_answer: str) -> list[AssertionModel]:
        """Extract the assertions from the RAG answer."""
        assert isinstance(rag_answer, str), "RAG answer must be a string"

        assertions = []
        lines = rag_answer.split("\n")

        for line in lines:
            if line.startswith("- "):
                parts = line[2:].split(" [")
                assertion = parts[0].strip()
                citations = [f"[{citation.strip()}" for citation in parts[1:]]
                assertion_model = AssertionModel(
                    assertion=assertion, citations=citations
                )
                assertions.append(assertion_model)

        LOGGER.info(f"📝 Extracted assertions: {assertions}")
        return assertions

    def _parse_response_into_queries(
        self, response_text: str, document_id: str, scenario: Scenario, tag: str
    ) -> list[Query]:
        """Parses the json in the generation into list of queries."""
        response_text = sanitise_response_text(response_text)
        generations = json.loads(response_text)["queries"]
        queries = []
        for q in generations:
            query = Query.from_response(
                text=q,
                document_id=document_id,
                model=scenario.model,
                prompt_template=scenario.prompt.prompt_template,
                tag=tag,
            )
            queries.append(query)
        return queries
