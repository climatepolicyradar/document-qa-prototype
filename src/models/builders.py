import re
from typing import Any, Tuple
from src.controllers.DocumentController import DocumentController
from src.models.data_models import (
    AssertionModel,
    Citation,
    EndToEndGeneration,
    ProcessedGenerationData,
    RAGRequest,
    RAGResponse,
    Scenario,
    _strip_inner_ai_monologue,
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
    page_number_cache: dict = {}

    @staticmethod
    def from_e2e_generation(e2e_generation: EndToEndGeneration):
        """
        Builds an E2E generation from an existing E2E generation.

        Does not set up the processed generation data or the citations. FOR NOW.
        """
        b = EndToEndGenerationBuilder()
        b.config = e2e_generation.config
        b.query = e2e_generation.rag_request.query
        b.scenario = e2e_generation.rag_request.as_scenario(DocumentController())
        b.retrieved_documents = (
            e2e_generation.rag_response.retrieved_documents
            if e2e_generation.rag_response is not None
            else []
        )
        b.answer = e2e_generation.get_answer()
        b.metadata = (
            e2e_generation.rag_response.metadata
            if e2e_generation.rag_response is not None
            else {}
        )
        return b

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

    def get_answer(self) -> str:
        """Returns the answer."""
        if not self.has_documents():
            return self.no_response_default_answer

        return self.answer

    def set_answer(self, answer: str):
        """Sets the answer. In doing so, extracts which sources were cited and populates the cited_documents and other_documents lists. Also breaks the answer up into assertions and generates the assertions list."""

        self.raw_answer = answer
        self.inner_monologue, self.answer = self._strip_inner_ai_monologue(
            self.raw_answer
        )
        self._setup_citations(self.answer)

        self.add_metadata("responded", not refused_answer(self.answer))

        self.assertions = [
            AssertionModel(
                assertion=assertion[0],
                citations=[
                    Citation(
                        citation_idx=idx,
                        cited=True,
                        text=self.retrieved_documents[idx]["page_content"]
                        if idx in range(0, len(self.retrieved_documents))
                        else "",
                    )
                    for idx in assertion[1]
                ],
            )
            for assertion in self._get_cited_document_indices_in_answer(self.answer)
        ]

        return self

    def _strip_inner_ai_monologue(self, text: str) -> Tuple[str, str]:
        """Extract the inner monologue from the RAG answer. Inner monologue is the text between #COT# and #/COT#. Returns (monologue, answer)"""

        try:
            return _strip_inner_ai_monologue(text)
        except Exception as e:
            logger.error(f"Error stripping inner monologue: {e}")
            self.add_metadata_list_item("errors", "Could not strip inner monologue")
            return ("", text)

    def _get_citation_label(self, doc: dict) -> str:
        """
        Returns a string label for the citation.

        Currently, the format is 'Pg. 1a' or 'ref. (a)', the latter when the page number is not present. Multiple references on one page will be Pg. 1b, Pg. 1c etc.
        """
        page_number = (
            doc["metadata"]["text_block_page"]
            if "text_block_page" in doc["metadata"]
            and doc["metadata"]["text_block_page"] is not None
            else None
        )

        if page_number is not None:
            if page_number not in self.page_number_cache:
                self.page_number_cache[page_number] = 1
            else:
                self.page_number_cache[page_number] += 1

            # Convert number of times we've seen this page number to alphabetical labels
            alpha_label = chr(
                96 + self.page_number_cache[page_number]
            )  # 'a' is 97 in ASCII

            return f"Pg. {page_number}{alpha_label}"

        if "NA" not in self.page_number_cache:
            self.page_number_cache["NA"] = 1
        else:
            self.page_number_cache["NA"] += 1
        alpha_label = chr(96 + self.page_number_cache["NA"])  # 'a' is 97 in ASCII
        return f"ref. ({alpha_label})"

    def _setup_citations(self, answer: str):
        """Sets up the citations."""
        # Extract the assertion sentences and the indices of the citations for them
        assertions_and_indices = self._get_cited_document_indices_in_answer(answer)

        if self.has_documents():
            unique_indices = set(
                [
                    index
                    for assertion_and_index in assertions_and_indices
                    for index in assertion_and_index[1]
                ]
            )
            # Set up other documents list and mark which ones are cited
            self.cited_documents = []
            self.other_documents = []
            self.page_number_cache = {}

            for i, doc in enumerate(self.retrieved_documents):
                # INDEXES ARE A NIGHTMARE. CHANGE AT YOUR PERIL.
                actual_idx = i

                doc["citation_idx"] = actual_idx
                cited = True if actual_idx in unique_indices else False
                doc["cited"] = cited
                doc["citation_label"] = self._get_citation_label(doc)

                if cited:
                    self.cited_documents.append(
                        Citation(
                            citation_idx=actual_idx,
                            cited=True,
                            text=doc["page_content"],
                        )
                    )

                if not cited:
                    self.other_documents.append(
                        Citation(
                            citation_idx=actual_idx,
                            cited=False,
                            text=doc["page_content"],
                        )
                    )

    def set_retrieved_documents(self, retrieved_documents: list[Any]):
        """Sets the retrieved documents."""
        self.retrieved_documents = [d.dict() for d in retrieved_documents]
        return self

    @property
    def no_response_default_answer(self) -> str:
        """
        Default answer when no documents are retrieved.

        This should trigger the answer refused logic in src.data_models.refused_answer
        and in auto-eval.
        """
        return "I cannot provide an answer based on the document."

    def hydrate_from_rag_chain_response(self, rag_chain_response: dict):
        """Pulls in data from the rag chain response"""
        self.query = rag_chain_response["query_str"]

        self.set_retrieved_documents(rag_chain_response["documents"])

        if len(rag_chain_response["documents"]) == 0:
            self.set_answer(self.no_response_default_answer)
        else:
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
            # pattern = r"\[(\d+(?:\s*,\s*\d+)*)\]"
            matches = re.findall(pattern, rag_answer)

            for sentence, citations in matches:
                citation_numbers = [int(c.strip()) for c in citations.split(",")]
                formatted_sentence = (
                    sentence.strip().lstrip("- ").lstrip(".").lstrip(",")
                )
                formatted_sentence = (
                    formatted_sentence[:1].upper() + formatted_sentence[1:]
                )
                results.append((formatted_sentence, citation_numbers))
        except Exception as e:
            logger.error(f"Error extracting cited document indices: {e}")
            self.add_metadata_list_item("errors", "Could not extract assertions")
            results = []

        return results
