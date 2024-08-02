import pandas as pd
import numpy as np

from collections import Counter
from typing import Optional, Union
from enum import Enum
from argilla import FeedbackDataset

from src.dataset_creation.annotation_dashboard.utils import transform_to_df, format_df
from src.evaluation.evaluator import Score


class AggregationStrategy(Enum):
    """Strategy for aggregating the human annotation responses"""

    MAJORITY = "majority"
    COMPLETE = "complete"
    AVERAGE = "average"


def argilla_to_scores(
    dataset: FeedbackDataset,
    strategy: Union[AggregationStrategy, str],
    questions: list[str],
) -> list[Score]:
    """
    Transforms the argilla feedback dataset into a list of scores

    Args:
        dataset (rg.FeedbackDataset): the dataset pulled from Argilla with the human annotations
        strategy (AggregationStrategy): the strategy to turn human annotations into scores
        questions (list[str]): list of questions to turn into scores

    Returns:
        list[Score]: list of Score objects with the aggregated annotations as the score


    Example usage:
    >>> feedback_from_hf = rg.FeedbackDataset.from_huggingface("ClimatePolicyRadar//annotation-responses-unece")
    >>> scores = argilla_to_scores(feedback_from_hf, "majority", ["toxicity", "faithfulness"])

    """
    if isinstance(strategy, str):
        strategy = AggregationStrategy(strategy)

    df = transform_to_df(dataset)

    scores = []

    for q in questions:
        temp_df = format_df(df.copy(), q)

        for q_id, question_group in temp_df.groupby("q_id"):
            if len(question_group) > 1:
                score = aggregate_scores(question_group, strategy)
            else:
                score = question_group["response"]

            if score is not None:
                scores.append(
                    Score(
                        score=score,
                        type=q,
                        name="human",
                        gen_uuid=q_id,
                    )
                )

    return scores


def aggregate_scores(
    group_df: pd.DataFrame, strategy: AggregationStrategy
) -> Optional[float]:
    """
    Taking the DataFrame of the questions responses and the strategy, outputs an aggregated score

    Args:
        group_df (pd.DataFrame): a dataframe with each row a separate response to the question
        strategy (AggregationStrategy): the strategy for aggregating the responses

    Returns:
        Optional[float]: returns a score between 0.0 and 1.0 (inclusive) or None in case
            the aggregation does not yield a decisive result. (E.g. there are 2 conflicting
            responses, and the strategy is "majority")

    TODO: the strategy also needs to specify "in which direction" it means the completeness.
        E.g. in the case of toxicity, does 0.0 or 1.0 have advantage (i.e. would we rather assume
        everything toxic with one response claiming so, or vice versa)

    TODO: we might want to be able to do annotator weighting, i.e. more reliable annotators
        having a higher weight than less reliable ones (might be derived from disagreement metrics)
    """

    responses = group_df["response"]

    if strategy == AggregationStrategy.AVERAGE:
        return np.mean(responses)
    elif strategy == AggregationStrategy.COMPLETE:
        return float(all(responses))
    elif strategy == AggregationStrategy.MAJORITY:
        majority_response, count = sorted(
            Counter(responses).items(), key=lambda x: x[1], reverse=True
        )[0]
        if count > len(responses) / 2:
            return majority_response
        else:
            return None
    else:
        raise NotImplementedError(f"Strategy {strategy.value} is unknown")
