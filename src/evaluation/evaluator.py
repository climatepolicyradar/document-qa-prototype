from src.models.data_models import EndToEndGeneration
from abc import ABC, abstractmethod
from src.logger import get_logger
from src.models.data_models import Score


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
        logger = get_logger(__name__)
        scores = []
        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(generation)
            except Exception as e:
                logger.error(
                    f"Error evaluating {generation.uuid} with {evaluator.NAME}/{evaluator.TYPE}: {e}"
                )
                continue
            if isinstance(score, list):
                scores.extend(score)
            elif isinstance(score, Score):
                scores.append(score)
        return scores
