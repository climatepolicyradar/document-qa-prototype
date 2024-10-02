from typing import Optional
from abc import ABC, abstractmethod

import re

from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.models.data_models import Score
from src.logger import get_logger
from src.online.inference import get_llm


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
        self.model = get_llm("gemini", model_name)

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

        if hasattr(response, "content"):
            result = response.content.strip()
        elif isinstance(response, str):
            result = response.strip()
        else:
            raise ValueError(f"Invalid response type: {type(response)}")

        if not result.isdigit():  # type: ignore
            raise ValueError(f"G-Eval score is not a digit: {result}")  # type: ignore

        score = self.response_postprocessor(result)

        if score is not None:
            normalised_score = (score - 1) / 4.0
            return Score(
                score=normalised_score,  # type: ignore
                success=self.get_success(normalised_score),
                type=self.TYPE,
                name=self.NAME,
                gen_uuid=generation.uuid,  # type: ignore
            )

    @abstractmethod
    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""
        pass

    @abstractmethod
    def get_success(self, score: float) -> bool:
        """Returns the success of the evaluator"""
        pass

    def response_postprocessor(self, response: str) -> Optional[int]:
        """Post-processes the response"""
        try:
            return int(response)
        except ValueError:
            try:
                LOGGER.warning(
                    f"Error processing response: {response}, trying to extract the score"
                )
                _digit_pattern = re.compile(r"(\d+)")
                match = _digit_pattern.search(response)
                if match is None:
                    return None
                return int(match.group(1))
            except Exception:
                LOGGER.error("Failed to extract the score from response")
                return None
