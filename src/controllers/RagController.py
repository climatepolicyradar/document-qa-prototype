from datetime import datetime
import json
from src.controllers.DocumentController import DocumentController
from src.controllers.ScenarioController import Scenario
from langchain_core.messages.base import BaseMessage
from cpr_data_access.models import BaseDocument
from typing import Optional, Union
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from src.controllers.VespaController import VespaController
from src.controllers.ObservabilityManager import ObservabilityManager

from src.logger import get_logger
from src.models.data_models import AssertionModel, EndToEndGeneration, Prompt, Query, RAGRequest, RAGResponse
from src.online.inference import get_llm
from src.dataset_creation.query_utils import render_document_text_for_llm, sanitise_response_text

from src.online.pipeline import rag_chain

from src.logger import get_logger


LOGGER = get_logger(__name__)

class RagController:
    def __init__(self):
        self.vespa = VespaController()
        self.observability = ObservabilityManager()
        
    def get_llm(self, type: str, model: str) -> Union[LLM, BaseChatModel]:  # type: ignore
        """
        Returns an instantiated LLM of the given type and model code.
        
        TODO: Move the get_llm underlying function to here for nicer encapsulation
        """
        return get_llm(type, model)
        
    def get_available_documents(self) -> list[BaseDocument]:
        """
        Passthrough method to Vespa controller. 
        
        Get the available documents in the Vespa database.
        """
        document_data = self.vespa.get_available_documents()
        dc = DocumentController()
        docs = [dc.create_base_document(doc["value"]) for doc in document_data["root"]["children"][0]["children"][0]["children"]]
        return docs
    
    def get_document_text(self, document_id: str):
        """
        Passthrough method to Vespa controller. 
        
        Get the text of a document from the Vespa database.
        """
        return self.vespa.get_document_text(document_id)
    

    def run_llm(self, scenario: Scenario, prompt_data: dict) -> str:
        """
        Run an LLM with the given scenario, prompt key, and prompt data.
        """
        llm = self.get_llm(scenario.generation_engine, scenario.model)
        prompt = scenario.prompt.prompt_content.render(prompt_data)
        
        LOGGER.info(f"🤔 Running prompt: {prompt}")
        return llm.invoke(prompt, config={"callbacks": [self.observability.get_tracing_callback()]})

    def generate_queries(
        self, 
        scenario: Scenario, 
        document: BaseDocument, 
        seed_queries: list[Query],
        tag: Optional[str] = None
    ) -> list[Query]:
        """
        Generate a set of queries using the given prompt template, model, document text and seed queries.
        """
        document_text = render_document_text_for_llm(document, scenario.model)
        
        llm = self.get_llm(scenario.generation_engine, scenario.model)

        prompt = scenario.prompt.prompt_content.render(
            product_queries=seed_queries, 
            document=document_text
        )
        
        response = llm.invoke(prompt, config={"callbacks": [self.observability.get_tracing_callback()]})
        
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
        
        return self._parse_response_into_queries(
            response.content, document.document_id, scenario, tag
        )
        
    def run_rag_pipeline(
        self,
        query: str, 
        scenario: Scenario,
    ) -> EndToEndGeneration:
        """
        Run the RAG pipeline for a given query and scenario.
        """
        LOGGER.info(f"***RUNNING RAG - query = `{query}`***")
        
        assert scenario.document is not None, "Scenario must have a document"
        
        start_time = datetime.now()
        
        llm = self.get_llm(scenario.generation_engine, scenario.model)
        
        LOGGER.info(f"🤔 Running document: {scenario.document.document_id}")
        
        rag_chain_with_source = rag_chain(
            llm=llm,
            retriever=self.vespa.retriever(scenario.document.document_id),
            citation_template=scenario.prompt.make_qa_prompt(),
            scenario=scenario
        )
        
        response = rag_chain_with_source.invoke({
            "query_str": query,
            "document_id": scenario.document.document_id
        }, config={"callbacks": [self.observability.get_tracing_callback()]})
        
        response_text = response['answer']
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
                        query=query
                        )
                    )
        
        return response
    
    def extract_assertions_from_answer(
        self, 
        answer: str, 
    ) -> list[AssertionModel]:
        """
        Extract the assertions from the RAG answer.
        """
        assertions = self._extract_assertions(answer)
        return assertions
    
    def highlight_key_quotes(self, scenario: Scenario, query: str, assertion: AssertionModel, context_str: str) -> str:
        """
        Highlight the key quotes in the given assertion.
        """
        highlight_prompt_key = "response/extract_key_quotes"
        
        highlight_scenario = Scenario(
            model=scenario.model,
            generation_engine=scenario.generation_engine,
            prompt=Prompt.from_template(highlight_prompt_key)
        )
        args = {
            "query_str": query,
            "assertion": assertion.assertion,
            "cited_sources": assertion.citations_as_string(),
            "context_str": context_str
        }
        
        LOGGER.info(f"🔍 Running highlight scenario with args: {args}")
        highlight_text = self.run_llm(highlight_scenario, args)
        LOGGER.info(f"🔍 Highlighted text: {highlight_text}")
        return highlight_text.content
        
        
    def _extract_assertions(self, rag_answer: str) -> list[str]:
        """
        Extract the assertions from the RAG answer.
        """
        assert isinstance(rag_answer, str), "RAG answer must be a string"
        
        assertions = []
        lines = rag_answer.split("\n")
        
        for line in lines:
            if line.startswith("- "):
                parts = line[2:].split(" [")
                assertion = parts[0].strip()
                citations = [f"[{citation.strip()}" for citation in parts[1:]]
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
                tag=tag
            )
            queries.append(query)
        return queries