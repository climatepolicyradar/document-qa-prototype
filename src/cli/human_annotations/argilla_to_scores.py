import click
import argilla as rg

from typing import Union
from tqdm import tqdm

from src.evaluation.human import argilla_to_scores
from src.logger import get_logger


LOGGER = get_logger(__name__)


@click.option(
    "--input-dataset",
    "-i",
    type=str,
    help="Input dataset huggingface repo",
    required=True,
    default="ClimatePolicyRadar/annotation-responses-unece",
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
    "--aggregation",
    "-g",
    type=str,
    help="The way to aggregate the data along the axis",
    required=True,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False),
    help="Output path for results",
    required=True,
    default="results.jsonl",
)
@click.command()
def evaluate(
    input_dataset: str,
    axis: Union[list[str], str],
    aggregation: str,
    output_path: str,
):
    """Evaluates the data along the specified axes"""
    if isinstance(axis, str):
        axis = [axis]

    feedback_from_hf = rg.FeedbackDataset.from_huggingface(input_dataset)

    scores = argilla_to_scores(feedback_from_hf, aggregation, axis)

    with open(output_path, "w") as f:
        for result in tqdm(scores):
            f.write(f"{result.model_dump_json()}\n")


if __name__ == "__main__":
    evaluate()
