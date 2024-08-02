import click
import json
import pandas as pd


from itertools import combinations

from src.evaluation.evaluator import Score
from src.logger import get_logger

LOGGER = get_logger(__name__)


@click.option(
    "--input-path",
    "-i",
    type=str,
    help="Path to the configuration file",
    required=True,
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    help="Output path for results",
    required=True,
    default="correlations.json",
)
@click.command()
def correlate(input_path: str, output_path: str):
    """Evaluates the data along the specified axes"""
    with open(input_path, "r") as f:
        scores = [Score.model_validate(json.loads(line)) for line in f.readlines()]

    all_axes = set(score.type for score in scores)
    all_correlations = {}

    for axis in all_axes:
        axis_scores = [score for score in scores if score.type == axis]
        df = pd.DataFrame([score.model_dump() for score in axis_scores])
        names = df["name"].unique()
        df = df.pivot_table(index="gen_uuid", columns="name", values="score")

        correlations = {}
        if len(names) > 1:
            LOGGER.info(f"Correlating {names} for {axis}")
            for n1, n2 in combinations(names, 2):
                _temp_df = df.copy()
                _temp_df.dropna(subset=[n1, n2], inplace=True)
                LOGGER.info(
                    f"Correlating {n1} and {n2} for {axis} in {_temp_df.shape[0]} samples"
                )
                correlations[f"{n1}-{n2}"] = _temp_df[[n1, n2]].corr()[n1][n2]

        all_correlations[axis] = correlations

    with open(output_path, "w") as f:
        f.write(json.dumps(all_correlations, indent=4))


if __name__ == "__main__":
    correlate()
