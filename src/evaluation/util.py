import catalogue
import importlib

from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import Evaluator
from src.logger import get_logger

LOGGER = get_logger(__name__)

evaluators = catalogue.create("src.evaluation", entry_points=True)


def get_evaluator(evaluator: str, kwargs: Optional[dict] = None) -> Optional[Evaluator]:
    if kwargs is None:
        kwargs = {}

    if evaluator not in evaluators:
        for path in Path("src/evaluation").rglob("*.py"):
            if path.stem == evaluator:
                if path.parent.stem == "evaluation":
                    module = "src.evaluation"
                else:
                    module = f"src.evaluation.{path.parent.stem}"
                try:
                    _ = importlib.import_module(f".{path.stem}", module)
                    break
                except ModuleNotFoundError:
                    continue

    if evaluator not in evaluators:
        LOGGER.warning(f"No parser found for {evaluator}")
        return None

    return evaluators.get(evaluator)(**kwargs)
