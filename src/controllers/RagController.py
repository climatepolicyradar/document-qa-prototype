import asyncio
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
from fastapi import WebSocket
from src.logger import get_logger
from src.models.data_models import (
    AssertionModel,
    EndToEndGeneration,
    Prompt,
    Query,
    RAGRequest,
    RAGResponse,
)
from src.online.inference import LLMTypes, get_llm
from src.dataset_creation.query_utils import (
    render_document_text_for_llm,
    sanitise_response_text,
)
import re

from src.online.pipeline import rag_chain, streamable_rag_chain
from src import config


LOGGER = get_logger(__name__)


class RagController:
    """Controller for RAG operations"""

    def __init__(self, observe: bool = True):
        self.vespa = VespaController()

        self.observability = ObservabilityManager()
        self.observe = False
        # TODO self.observe = observe

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

        LOGGER.info(f"🤔 Running {scenario.model}")

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
            retriever=self.vespa.retriever(
                scenario.document.document_id, scenario.top_k_retrieval_results or 6
            ),  # type: ignore
            citation_template=scenario.prompt.make_qa_prompt(),
            scenario=scenario,
        )

        response = rag_chain_with_source.invoke(
            {
                "query_str": query,
                "document_id": scenario.document.document_id,
                "document_metadata_context_str": f"'{scenario.document.document_name}' pub. {scenario.document.document_metadata.publication_ts} (country:{scenario.document.document_metadata.geography})",
            },
        )

        response_text = response["answer"]
        LOGGER.info(f"Response: {response_text}")

        end_time = datetime.now()
        duration = end_time - start_time
        LOGGER.info(f"Duration: {duration}")

        metadata = {}
        try:
            metadata["assertions"] = self.extract_assertions_from_answer(response_text)
        except Exception as e:
            LOGGER.error(f"Error extracting assertions: {e}")
            metadata["errors"] = ["Could not extract assertions"]

        response = EndToEndGeneration(
            config=scenario.get_config(),
            rag_request=RAGRequest.from_scenario(query, scenario),
            rag_response=RAGResponse(
                text=response_text,
                retrieved_documents=[d.dict() for d in response["documents"]],
                query=query,
                metadata=metadata,
            ),
        )

        return response

    async def stream_rag_pipeline(
        self, query: str, scenario: Scenario, websocket: WebSocket
    ):
        """
        Stream the RAG pipeline for a given query and scenario.

        This is a streaming RAG pipeline that sends the RAG response to the client in chunks.
        """
        LOGGER.info(f"***STREAMING RAG - query = `{query}`***")

        assert scenario.document is not None, "Scenario must have a document"

        start_time = datetime.now()

        LOGGER.info(f"🤔 Running generation engine: {scenario.generation_engine}")
        LOGGER.info(f"🤔 Running model: {scenario.model}")

        llm = self.get_llm(scenario.generation_engine, scenario.model)

        LOGGER.info(f"🤔 Running document: {scenario.document.document_id}")

        rag_chain_with_source = streamable_rag_chain(
            llm=llm,
            retriever=self.vespa.retriever(
                scenario.document.document_id, scenario.top_k_retrieval_results or 6
            ),  # type: ignore
            citation_template=scenario.prompt.make_qa_prompt(),
            scenario=scenario,
        )

        response_text = ""
        documents = []
        for event in rag_chain_with_source.stream(
            {
                "query_str": query,
                "document_id": scenario.document.document_id,
                "document_metadata_context_str": f"'{scenario.document.document_name}' pub. {scenario.document.document_metadata.publication_ts} (country:{scenario.document.document_metadata.geography})",
            }
        ):
            print(event, flush=True)

            await websocket.send_text(event.strip())
            await asyncio.sleep(0)  # Yield control back to the event loop

            response_text += event

        LOGGER.info(f"Response: {response_text}")

        end_time = datetime.now()
        duration = end_time - start_time
        LOGGER.info(f"Duration: {duration}")

        metadata = {}
        try:
            metadata["assertions"] = self.extract_assertions_from_answer(response_text)
        except Exception as e:
            LOGGER.error(f"Error extracting assertions: {e}")
            metadata["errors"] = ["Could not extract assertions"]

        response = EndToEndGeneration(
            config=scenario.get_config(),
            rag_request=RAGRequest.from_scenario(query, scenario),
            rag_response=RAGResponse(
                text=response_text,
                retrieved_documents=[d.dict() for d in documents],
                query=query,
                metadata=metadata,
            ),
        )

        return response

    def execute_no_answer_flow(self, result: EndToEndGeneration) -> EndToEndGeneration:
        """Used to generate the information for no answer flows"""

        LOGGER.info(
            f"🔍 System did not respond to the user query: {result.rag_request.query}: {result.get_answer()}"
        )
        scenario = Scenario(
            prompt=Prompt.from_template("response/summarise_simple"),
            model="mistral-nemo",
            generation_engine=LLMTypes.VERTEX_AI.value,
        )

        if result.rag_response is None:
            raise ValueError("RAG response is None")

        summary = self.run_llm(
            scenario, {"query_str": result.rag_response.retrieved_passages_as_string()}
        )

        LOGGER.info(f"🔍 System summarised the query: {summary}")
        result.rag_response.add_metadata("no_answer_summary", summary)
        result.rag_response.add_metadata(
            "no_answer_assertions", self.extract_assertions_from_answer(summary)
        )
        return result

    def extract_assertions_from_answer(
        self,
        answer: str,
    ) -> list[AssertionModel]:
        """Extract the assertions from the RAG answer."""
        assertions = self._extract_assertions(answer)
        return assertions

    async def highlight_key_quotes(
        self,
        query: str,
        assertion: AssertionModel,
        retrieved_documents: list[dict],
    ) -> str:
        """
        Highlight the key quotes in the given assertion.

        Assumes that the given assertion model has only one citation. pre-process using .to_atomic_assertions if that is not the case.
        """
        highlight_prompt_key = "response/extract_key_quotes"

        highlight_scenario = Scenario(
            model="mistral-nemo",
            generation_engine="vertexai",
            prompt=Prompt.from_template(highlight_prompt_key),
        )
        args = {
            "query_str": query,
            "assertion": assertion.assertion,
            "context_str": retrieved_documents[int(assertion.citations[0]) - 1][
                "page_content"
            ],
        }

        LOGGER.info(f"🔍 Running {assertion} highlight scenario with args: {args}")
        highlight_text = self.run_llm(highlight_scenario, args)
        LOGGER.info(f"🔍 Highlighted text: {highlight_text}")
        return highlight_text

    def _extract_assertions(self, rag_answer: str) -> list[AssertionModel]:
        """Extract the assertions from the RAG answer."""
        assert isinstance(rag_answer, str), "RAG answer must be a string"

        assertions = []

        pattern = r"(.*?)\s*\[([\d,\s]+)\]"  # Looking for [x], [x,y], [x,y,z] etc
        matches = re.findall(pattern, rag_answer)

        results = []
        for sentence, citations in matches:
            citation_numbers = [c.strip() for c in citations.split(",")]
            results.append((sentence.strip(), citation_numbers))

        LOGGER.info(f"🔍 Extracted {len(results)} sentences with citations")

        # extracted_sentences = extract_sentences_with_citations(rag_answer)
        # LOGGER.debug(f"🔢 Extracted sentences: {extracted_sentences}")

        assert len(results) > 0, "No sentences with citations found"

        for line in results:
            print(line)
            assertion = line[0]
            citations = line[1]
            assertion_model = AssertionModel(assertion=assertion, citations=citations)
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
