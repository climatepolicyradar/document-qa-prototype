import click
import pandas as pd

from typing import Union
from tqdm import tqdm

from src.evaluation.evaluator import MultiAxisEvaluator, Score
from src.models.data_models import EndToEndGeneration
from src.evaluation.util import get_evaluator
from src.logger import get_logger


LOGGER = get_logger(__name__)


@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True),
    help="Path to the configuration file",
    required=True,
)
@click.option(
    "--axis",
    "-a",
    type=str,
    help="Name of the axes to evaluate the data along",
    required=True,
    multiple=True,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False),
    help="Output path for results",
    required=True,
    default="results.jsonl",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    help="The maximum number of generations to evaluate",
    required=False,
    default=-1,
)
@click.option(
    "--start",
    "-s",
    type=int,
    help="The start index of generations if you want to skip any",
    required=False,
    default=0,
)
@click.command()
def evaluate(
    input_path: str,
    axis: Union[list[str], str],
    output_path: str,
    limit: int,
    start: int,
):
    """Evaluates the data along the specified axes"""
    generations_df = pd.read_json(input_path, lines=True)

    end_to_end_generations = [
        EndToEndGeneration.model_validate(generation)
        for generation in tqdm(
            generations_df["generation"].tolist()[start : start + limit],
            total=limit,
            desc="Validating generations",
        )
    ]

    if isinstance(axis, str):
        axis = [axis]

    _evaluators = [get_evaluator(name) for name in axis]
    eval = MultiAxisEvaluator([e for e in _evaluators if e is not None])

    all_results: list[Score] = []
    for e2e_gen in tqdm(end_to_end_generations, total=len(end_to_end_generations)):
        try:
            results = eval.evaluate(e2e_gen)
        except ValueError as e:
            results = []
            LOGGER.error(f"Error evaluating generation {e2e_gen.uuid}: {e}")
        all_results += results

    with open(output_path, "w") as f:
        for result in tqdm(all_results):
            if result is not None:
                f.write(f"{result.model_dump_json()}\n")


if __name__ == "__main__":
    evaluate()
