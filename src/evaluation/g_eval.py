from typing import Optional
from abc import ABC, abstractmethod

from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.controllers.RagController import RagController
from src.evaluation.evaluator import Score
from src.logger import get_logger


LOGGER = get_logger(__name__)


class GEval(Evaluator, ABC):
    """
    G-Eval abstract class

    This is a base class for the G-Eval evaluators. It provides the common methods, pre- and
    post-processing steps for the evaluators.

    paper: https://arxiv.org/pdf/2303.16634
    repo: https://github.com/nlpyang/geval
    """

    TYPE = ""
    NAME = ""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model = RagController().get_llm("gemini", model_name)

    def evaluate(
        self, generation: EndToEndGeneration, prompt: Optional[str] = None
    ) -> Optional[Score]:
        """Evaluates the generation"""
        self._validate_generation(generation)

        if generation.rag_response is None:
            return None

        if prompt is None:
            prompt = self.get_prompt(generation)
        response = self.model.invoke(prompt)

        result = response.content.strip()

        if not result.isdigit():  # type: ignore
            raise ValueError(f"G-Eval score is not a digit: {response.content}")  # type: ignore

        score = self.response_postprocessor(response)

        if score is not None:
            return Score(
                score=(score - 1) / 4.0,  # type: ignore
                type=self.TYPE,
                name=self.NAME,
                gen_uuid=generation.uuid,  # type: ignore
            )

    @abstractmethod
    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""
        pass

    def response_postprocessor(self, response) -> Optional[int]:
        """Post-processes the response"""
        return int(response.content)
