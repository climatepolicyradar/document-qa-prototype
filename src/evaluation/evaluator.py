from src.models.data_models import EndToEndGeneration
from pydantic import BaseModel, confloat
from typing import Optional
from abc import ABC, abstractmethod


class Score(BaseModel):
    """Score model"""

    score: confloat(ge=0.0, le=1.0)  # type: ignore
    type: str
    name: str
    gen_uuid: str
    comments: Optional[list[str]] = None


class Evaluator(ABC):
    """Evaluator interface"""

    TYPE = None
    NAME = None

    @abstractmethod
    def evaluate(self, data: EndToEndGeneration) -> Score:
        """Evaluates the data"""
        pass

    def _validate_generation(self, generation: EndToEndGeneration) -> None:
        assert generation.uuid is not None


class MultiAxisEvaluator(Evaluator):
    """
    Multi-axis evaluator.

    Uses multiple evaluators to evaluate the data.
    """

    def __init__(self, evaluators: list[Evaluator]):
        self.evaluators = evaluators

    def evaluate(self, generation: EndToEndGeneration) -> list[Score]:
        """Evaluates along the axes of the evaluators"""
<<<<<<< HEAD
        
        return [evaluator.evaluate(generation) for evaluator in self.evaluators]
=======
        scores = []
        for evaluator in self.evaluators:
            score = evaluator.evaluate(generation)
            if isinstance(score, list):
                scores.extend(score)
            elif isinstance(score, Score):
                scores.append(score)
        return scores
>>>>>>> main
