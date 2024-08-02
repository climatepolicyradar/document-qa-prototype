import click
import pandas as pd
import catalogue
import importlib

from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import Evaluator
from typing import Union
from tqdm import tqdm
from catalogue import Registry

from src.evaluation.evaluator import MultiAxisEvaluator, Score
from src.models.data_models import EndToEndGeneration
from src.logger import get_logger

evaluators = catalogue.create("src.evaluation", entry_points=True)

LOGGER = get_logger(__name__)

class EvaluationController():
    evaluators: Registry 
    instantiated: list[Evaluator]
    
    def __init__(self):
        self.evaluators = evaluators
        self.instantiated = [
            self.get_evaluator("g_eval_policy"), 
            self.get_evaluator("g_eval_faithfulness"), 
            self.get_evaluator("vectara")
        ] # The registry doesn't instantiate the evaluators properly, so load them separately
    
    def get_all_evaluators(self):
        return self.instantiated

    def get_evaluator(self, evaluator: str, kwargs: Optional[dict] = None) -> Evaluator:
        if kwargs is None:
            kwargs = {}

        if evaluator not in self.evaluators:
            for path in Path("src/evaluation").rglob("*.py"):
                if path.stem == evaluator:
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

    def evaluate(self, result: EndToEndGeneration, evaluator: str, eval_kwargs: Optional[dict] = None):
        evaluator = self.get_evaluator(evaluator, eval_kwargs)
        return evaluator.evaluate(result)
     
    def evaluate_all(self, result: EndToEndGeneration):
        eval: MultiAxisEvaluator = MultiAxisEvaluator(self.instantiated)
        return eval.evaluate(result)
