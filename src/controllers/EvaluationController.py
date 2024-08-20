import asyncio
import catalogue
import importlib
import json

from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import Evaluator
from catalogue import Registry

from src.evaluation.evaluator import MultiAxisEvaluator
from src.models.data_models import EndToEndGeneration
from src.logger import get_logger

evaluators = catalogue.create("src.evaluation", entry_points=True)

LOGGER = get_logger(__name__)


class EvaluationController:
    """Controller for managing and executing evaluations."""

    evaluators: Registry
    instantiated: list[Evaluator]

    def __init__(self):
        self.evaluators = evaluators
        self.instantiated = [
            self.get_evaluator("formatting"),
            self.get_evaluator("g_eval_policy"),
            self.get_evaluator("g_eval_faithfulness"),
            self.get_evaluator("system_response"),
            self.get_evaluator("patronus_lynx"),
            self.get_evaluator("vectara"),
        ]  # The registry doesn't instantiate the evaluators properly, so load them separately

    def set_evaluators(self, evals: list[str]):
        """Set a custom set of evaluators to use"""
        self.instantiated = [self.get_evaluator(e) for e in evals]

    def get_all_evaluators(self):
        """Retrieve all registered evaluators."""
        return self.instantiated

    def get_evaluator(self, evaluator: str, kwargs: Optional[dict] = None) -> Evaluator:
        """
        Get a specific evaluator by name.

        Args:
            evaluator (str): The name of the evaluator to retrieve.
            kwargs (Optional[dict]): Additional keyword arguments for the evaluator.

        Returns:
            Evaluator: The requested evaluator instance.
        """
        if kwargs is None:
            kwargs = {}

        if evaluator not in self.evaluators:
            for path in Path("src/evaluation").rglob("*.py"):
                if path.stem.strip() == evaluator:
                    try:
                        _ = importlib.import_module(
                            f".{path.stem}", f"src.evaluation.{path.parent.stem}"
                        )
                        break
                    except ModuleNotFoundError:
                        continue

        if evaluator not in self.evaluators:
            LOGGER.warning(f"No parser found for {evaluator}")
            raise ValueError(f"No parser found for {evaluator}")

        return self.evaluators.get(evaluator)(**kwargs)

    def evaluate(
        self,
        result: EndToEndGeneration,
        evaluator: str,
        eval_kwargs: Optional[dict] = None,
    ):
        """
        Evaluate the given result using the specified evaluator.

        Args:
            result (EndToEndGeneration): The generation to evaluate.
            evaluator (str): The name of the evaluator to use.
            eval_kwargs (Optional[dict]): Additional keyword arguments for the evaluator.

        Returns:
            The evaluation result from the specified evaluator.
        """
        eval_inst = self.get_evaluator(evaluator, eval_kwargs)
        return eval_inst.evaluate(result)

    def evaluate_system_response(self, result: EndToEndGeneration):
        """Evaluate the system response."""
        return self.evaluate(result, "system_response")

    def evaluate_all(self, result: EndToEndGeneration):
        """
        Evaluate the given result using all instantiated evaluators.

        Args:
            result (EndToEndGeneration): The generation to evaluate.

        Returns:
            The evaluation results from all instantiated evaluators.
        """
        eval: MultiAxisEvaluator = MultiAxisEvaluator(self.instantiated)

        return eval.evaluate(result)

    async def evaluate_async(self, result: EndToEndGeneration):
        """Evaluate the given result using all instantiated evaluators in parallel. Coupled to API usage. Needs a good refactor really."""
        if result is None or result.rag_response is None:
            LOGGER.warning("Result is None, skipping evaluation")
            return None

        async def process_eval_async(gen: EndToEndGeneration, evaluator: str):
            result = self.evaluate(gen, evaluator)
            if result is not None:
                return json.dumps(result.model_dump())
            else:
                return None

        # need to dump to string for gather to work
        result = await asyncio.gather(
            *[
                process_eval_async(result, evaluator)
                for evaluator in [
                    "formatting",
                    "g_eval_policy",
                    "g_eval_faithfulness",
                    "system_response",
                    "patronus_lynx",
                    "vectara",
                ]
            ]  # type: ignore
        )
        filtered_results = [r for r in result if r is not None]
        result = [json.loads(r) for r in filtered_results]  # type: ignore
        return result

    def did_system_respond(self, result: EndToEndGeneration):
        """Check if the system responded to the user query."""
        sys_eval = self.get_evaluator("system_response")
        eval_result = sys_eval.evaluate(result)
        if eval_result.score == 0:
            return False
        return True
