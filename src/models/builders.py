import re
from typing import Any, Tuple
from src.models.data_models import (
    AssertionModel,
    Citation,
    EndToEndGeneration,
    ProcessedGenerationData,
    RAGRequest,
    RAGResponse,
    Scenario,
    refused_answer,
)
from src.logger import get_logger

logger = get_logger(__name__)


class EndToEndGenerationBuilder:
    """Builder for EndToEndGeneration objects."""

    config: dict = {}
    query: str = ""
    raw_answer: str = ""
    answer: str = ""
    inner_monologue: str = ""
    scenario: Scenario
    retrieved_documents: list[dict] = []
    cited_documents: list[Citation] = []
    other_documents: list[Citation] = []
    metadata: dict = {}

    def __call__(self):
        """
        Returns the final EndToEndGeneration object.

        Use like this:
        ```
        gen_model = EndToEndGenerationBuilder().set_scenario(scenario).set_retrieved_documents(retrieved_documents).set_answer(answer)

        return gen_model() # this function
        ```
        """
        return EndToEndGeneration(
            config=self.config,
            rag_request=RAGRequest.from_scenario(self.query, self.scenario),
            rag_response=RAGResponse(
                text=self.answer,
                retrieved_documents=self.retrieved_documents,
                query=self.query,
                metadata=self.metadata,
            ),
            processed_generation_data=ProcessedGenerationData(
                final_answer=self.answer,
                inner_monologue=self.inner_monologue,
                assertions=self.assertions,
                cited_documents=self.cited_documents,
                other_documents=self.other_documents,
            ),
        )

    def set_scenario(self, scenario: Scenario):
        """Sets the scenario."""
        self.scenario = scenario
        return self

    def has_documents(self) -> bool:
        """Returns True if the generation has documents."""
        return len(self.retrieved_documents) > 0

    def set_answer(self, answer: str):
        """Sets the answer. In doing so, extracts which sources were cited and populates the cited_documents and other_documents lists. Also breaks the answer up into assertions and generates the assertions list."""

        self.raw_answer = answer
        self.inner_monologue, self.answer = self._strip_inner_ai_monologue(
            self.raw_answer
        )

        assertions_and_indices = self._setup_citations(self.answer)

        self.add_metadata("responded", not refused_answer(self.answer))

        self.assertions = [
            AssertionModel(assertion=assertion[0], citations=self.cited_documents)
            for assertion in assertions_and_indices
        ]

        return self

    def _strip_inner_ai_monologue(self, text: str) -> Tuple[str, str]:
        """Extract the inner monologue from the RAG answer. Inner monologue is the text between #COT# and #/COT#. Returns (monologue, answer)"""

        try:
            # Some quick LLMS ARE UNRULY CHILDREN checks
            if "# /COT#" in text:
                text = text.replace("# /COT#", "#/COT#")
            if "#/COT #" in text:
                text = text.replace("#/COT #", "#/COT#")

            if "#COT#" in text and "#/COT#" in text:
                return (
                    text.split("#COT#")[1].split("#/COT#")[0],
                    text.split("#/COT#")[1],
                )
            else:
                return ("", text)
        except Exception as e:
            logger.error(f"Error stripping inner monologue: {e}")
            self.add_metadata_list_item("errors", "Could not strip inner monologue")
            return ("", text)

    def _setup_citations(self, answer: str):
        """Sets up the citations."""
        # Extract the assertion sentences and the indices of the citations for them
        assertions_and_indices = self._get_cited_document_indices_in_answer(answer)

        if self.has_documents():
            self.cited_documents = [
                Citation(
                    citation_idx=int(index),
                    cited=True,
                    text=self.retrieved_documents[int(index)]["page_content"],
                )
                for assertion_and_index in assertions_and_indices
                for index in assertion_and_index[1]
            ]
            for i, doc in enumerate(self.retrieved_documents):
                if i not in [
                    citation.citation_idx for citation in self.cited_documents
                ]:
                    self.other_documents.append(
                        Citation(
                            citation_idx=i,
                            cited=False,
                            text=doc["page_content"],
                        )
                    )
        return assertions_and_indices

    def set_retrieved_documents(self, retrieved_documents: list[Any]):
        """Sets the retrieved documents."""
        self.retrieved_documents = [d.dict() for d in retrieved_documents]
        return self

    def hydrate_from_rag_chain_response(self, rag_chain_response: dict):
        """Pulls in data from the rag chain response"""
        self.query = rag_chain_response["query_str"]

        self.set_retrieved_documents(rag_chain_response["documents"])
        self.set_answer(rag_chain_response["answer"])

        return self

    def add_metadata(self, key: str, value: Any):
        """Adds a key-value pair to the metadata dictionary."""
        self.metadata[key] = value
        return self

    def add_metadata_list_item(self, key: str, value: Any):
        """Adds a value to a list in the metadata dictionary."""
        if self.metadata.get(key) is None:
            self.metadata[key] = []
        self.metadata[key].append(value)
        return self

    def get_metadata(self, key: str) -> Any:
        """Gets a value from the metadata dictionary."""
        return self.metadata.get(key)

    def set(self, property_name: str, value: Any) -> "EndToEndGenerationBuilder":
        """
        Set a property of the builder.

        Args:
            property_name (str): The name of the property to set.
            value (Any): The value to set the property to.

        Returns:
            EndToEndGenerationBuilder: The builder instance for method chaining.

        Raises:
            AttributeError: If the property doesn't exist.
        """
        if not hasattr(self, property_name):
            raise AttributeError(f"🚫 Property '{property_name}' does not exist.")

        setattr(self, property_name, value)
        return self

    def _get_cited_document_indices_in_answer(self, rag_answer: str):
        results = []
        try:
            """Get the indices of the cited documents in the answer."""
            pattern = r"(.*?)\s*\[([\d,\s]+)\]"  # Looking for [x], [x,y], [x,y,z] etc
            matches = re.findall(pattern, rag_answer)

            for sentence, citations in matches:
                citation_numbers = [int(c.strip()) for c in citations.split(",")]
                formatted_sentence = (
                    sentence.strip().lstrip("- ").lstrip(".").lstrip(",")
                )
                formatted_sentence = formatted_sentence.capitalize()
                results.append((formatted_sentence, citation_numbers))
        except Exception as e:
            logger.error(f"Error extracting cited document indices: {e}")
            self.add_metadata_list_item("errors", "Could not extract assertions")
            results = []

        return results
