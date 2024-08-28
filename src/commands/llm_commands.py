from abc import abstractmethod

from src.models.data_models import EndToEndGeneration, Prompt, Scenario
from src.online.inference import LLMTypes


class LLMCommand:
    """Base class for LLM commands."""

    def __init__(self, rag_controller):
        """They all use a RAG controller"""
        self.rag_controller = rag_controller

    @abstractmethod
    def __call__(self, end_to_end_generation: EndToEndGeneration, **kwargs) -> str:
        """Make it a callable class"""
        pass


class SummariseDocuments(LLMCommand):
    """Summarises the documents retrieved by the RAG controller."""

    def __call__(self, end_to_end_generation: EndToEndGeneration, **kwargs):
        """Call the command"""
        scenario = Scenario(
            prompt=Prompt.from_template("response/summarise_simple"),
            model="mistral-nemo",
            generation_engine=LLMTypes.VERTEX_AI.value,
        )

        if not end_to_end_generation.has_documents():
            return ""

        summary = self.rag_controller.run_llm(
            scenario,
            {"query_str": end_to_end_generation.get_documents()},
        )

        return summary


class GetTopicsFromText(LLMCommand):
    """Gets the topics from the text retrieved by the RAG controller."""

    def _scenario(self) -> Scenario:
        return Scenario(
            prompt=Prompt.from_template(
                "response/generate_topics_from_retrieved_documents"
            ),
            model="mistral-nemo",
            generation_engine=LLMTypes.VERTEX_AI.value,
        )

    def __call__(self, end_to_end_generation: EndToEndGeneration, **kwargs):
        """Call the command"""

        if not end_to_end_generation.has_documents():
            return []

        retrieved_passages_joined = " ".join(
            [doc["page_content"] for doc in end_to_end_generation.get_documents()]
        )

        return self.process_text(retrieved_passages_joined)

    def process_text(self, text: str) -> list[str]:
        """Process the text to remove unwanted characters."""

        topics = self.rag_controller.run_llm(self._scenario, {"context_str": text})

        try:
            topics_list = self._process_extracted_topics(topics)
        except Exception:
            topics_list = []

        return topics_list

    def _process_extracted_topics(self, topics: str) -> list[str]:
        """Process the extracted topics from the no answer flow."""

        topics_split = topics.lower().replace("-–", "").strip().split("\n")

        return topics_split
