from enum import Enum
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.evaluation.evaluator import Score
from src.logger import get_logger


LOGGER = get_logger(__name__)


class EnsemblingStrategy(Enum):
    """Strategy types for ensembling evaluator scores"""

    ALL = "all"
    ANY = "any"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"


def ensemble(
    scores: list[Score],
    evaluator_names: list[str],
    strategy: EnsemblingStrategy,
    weights: Optional[dict[str, float]] = None,
) -> list[Score]:
    """
    Ensembles the scores using the specified strategy.

    Creates a list of new scores with the same type as the initial scores, and a new
    metric name, which is of the structure: "ensemble:<name_1>-<name_2>...<name_n>".

    This works with an arbitrary number of metric names.

    Args:
        scores (list[Score]): the initial scores to perform ensembling on
        evaluator_names (list[str]): the names of the evaluators to ensemble (they must be in the scores)
        strategy (EnsemblingStrategy): the strategy to use for ensembling (all, any, average or weighted average)
        weights: (Optional[dict[str, float]]): in case of weighted average the map of metrics name to weight

    Raises:
        AssertionError: if there are multiple types of scores provided. This assumes that the scores are prefiltered.

    Returns:
        list[Score]: the list of new Scores with ensembled score, and the new ensemble metric name
    """
    assert (
        len(set(score.type for score in scores)) == 1
    ), f"There are multiple types in the provided scores: {set(score.type for score in scores)}"

    ensemble_function = get_ensemble_function(strategy, weights)

    df = pd.DataFrame([score.model_dump() for score in scores])
    metric_type = df["type"].unique()[0]
    df = df.pivot_table(index="gen_uuid", columns="name", values="score")

    ensemble_name = f"ensemble:{strategy.value}:{'-'.join(evaluator_names)}"

    df[ensemble_name] = df[evaluator_names].apply(
        lambda x: ensemble_function(x), axis=1
    )
    df[ensemble_name] = df[ensemble_name].astype(float)

    return df.apply(
        lambda row: Score(
            gen_uuid=row.name,
            name=ensemble_name,
            score=row[ensemble_name],
            type=metric_type,
        ),
        axis=1,
    ).tolist()


def get_ensemble_function(
    strategy: EnsemblingStrategy, weights: Optional[dict[str, float]] = None
) -> Callable:
    """Returns the ensemble function for the specified strategy to be applied on the dataframe rows"""

    # TODO: need to make thresholds configurable
    if strategy == EnsemblingStrategy.ALL:
        return lambda row: all([v > 0.5 for v in row.values])
    elif strategy == EnsemblingStrategy.ANY:
        return lambda row: any([v > 0.5 for v in row.values])
    elif strategy == EnsemblingStrategy.AVERAGE:
        return lambda row: np.average(row.values)
    elif strategy == EnsemblingStrategy.WEIGHTED_AVERAGE:
        if weights is not None:
            return lambda row: np.average(
                row.values,
                weights=[weights.get(name) for name in row.index],  #  type: ignore
            )
        else:
            raise ValueError("Weights must be provided for weighted average ensembling")
    else:
        raise NotImplementedError(f"Unknown strategy {strategy.value}")
