from datetime import datetime
import json
import random

from fastapi import HTTPException
from src.commands.llm_commands import GetTopicsFromText, SummariseDocuments
from src.controllers.ScenarioController import Scenario
from cpr_data_access.models import BaseDocument
from typing import Optional
from langchain_core.messages.base import BaseMessage
from langchain_core.language_models.base import BaseLanguageModel
from src.controllers.VespaController import VespaController
from src.controllers.ObservabilityManager import ObservabilityManager
from src.controllers.GuardrailController import GuardrailController

from src.logger import get_logger
from src.models.builders import EndToEndGenerationBuilder
from src.models.data_models import (
    EndToEndGeneration,
    Query,
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
        self.guardrails = GuardrailController()
        self.observe = False
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

        TODO extract to LLMCommand
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
        return_early_on_guardrail_failure: bool = False,
    ) -> EndToEndGeneration:
        """
        Run the RAG pipeline for a given query and scenario.

        :param query: user query
        :param scenario: scenario to run
        :param return_early_on_guardrail_failure: Whether to return early if the input
            guardrail fails. Defaults to False
        :return EndToEndGeneration: object containing request and response information
            and metadata
        """

        LOGGER.info(f"***RUNNING RAG - query = `{query}`{scenario.document}***")

        assert scenario.document is not None, "Scenario must have a document"

        generation = (
            EndToEndGenerationBuilder()
            .set_scenario(scenario)
            .add_metadata("guardrails", {})
            .add_metadata("guardrails_failed", False)
        )

        start_time = datetime.now()

        LOGGER.info(f"🔍 Running guardrails for query: {query}")
        guardrail_result = self.guardrails.validate(query)
        LOGGER.info(f"🔍 Guardrail result: {guardrail_result}")

        if guardrail_result.overall_result:
            LOGGER.info(f"🔍 Guardrail passed for query: {query}")
        else:
            LOGGER.info(f"🔍 Guardrail failed for query: {query}")
            if return_early_on_guardrail_failure:
                raise HTTPException(
                    status_code=400,
                    detail="We could not process this query because it failed our safety checks.",
                )

        llm = self.get_llm(scenario.generation_engine, scenario.model)

        rag_chain_with_source = rag_chain(
            llm=llm,
            retriever=self.vespa.retriever(
                scenario.document.document_id,  # type: ignore
                scenario.top_k_retrieval_results or 6,
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

        generation.hydrate_from_rag_chain_response(response)

        LOGGER.info(f"🔍 Running guardrrails on response: {generation.get_answer()}")
        guardrail_result = self.guardrails.validate(generation.get_answer())
        LOGGER.info(f"🔍 Guardrail result: {guardrail_result}")

        generation.add_metadata("guardrails", guardrail_result)
        generation.add_metadata(
            "guardrails_failed", not guardrail_result.overall_result
        )

        LOGGER.info(f"Response: {generation.raw_answer}")
        LOGGER.info(f"Duration: {datetime.now()-start_time}")

        return generation()

    def execute_no_answer_flow(self, result: EndToEndGeneration) -> EndToEndGeneration:
        """Used to generate the information for no answer flows"""

        LOGGER.info(
            f"🔍 System did not respond to the user query: {result.rag_request.query}: {result.get_answer()}"
        )

        if result.rag_response is None:
            raise ValueError("RAG response is None")

        summary = SummariseDocuments(rag_controller=self)(result)
        topics_list = GetTopicsFromText(rag_controller=self)(result)

        LOGGER.info(f"🔍 System summarised the query: {summary}")
        result.add_metadata("no_answer_summary", summary)
        result.add_metadata("no_answer_topics", topics_list)

        return result

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
