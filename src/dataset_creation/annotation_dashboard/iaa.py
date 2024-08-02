from copy import deepcopy
from functools import reduce
from typing import Callable, Literal, Optional
import argilla as rg
import pandas as pd
import math
import numpy as np

from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from collections import Counter
from scipy.stats import skew
from argilla import FeedbackDataset
from argilla.client.feedback.metrics import AgreementMetric
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

from src.dataset_creation.annotation_dashboard.utils import (
    transform_to_df,
    format_df,
    _filter_dataset_by_response,
    _filter_dataset_by_users,
    default_reject_responses,
    likert_questions,
)
from src.logger import get_logger


LOGGER = get_logger(__name__)


def calculate_iaa_table(
    dataset: FeedbackDataset,
    questions: list[str],
    engine: Literal["argilla", "nltk"] = "nltk",
    exclude_dont_know: bool = False,
    rejected_users: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Calculates the inter-annotator agreement for a list of questions in a dataset.

    Args:
        dataset (FeedbackDataset): The rg.FeedbackDataset to calculate the IAA on
        questions (list[str]): The list of questions to calculate the IAA for
        engine (Literal["argilla", "nltk"], optional): The engine to use. Defaults to "nltk". NOTE: Argilla is deprecated
        exclude_dont_know (bool, optional): Whether to exclude the "DONT_KNOW" / "NOT_APPLICABLE" responses. Defaults to False.
        rejected_users (Optional[list[str]], optional): A list of users to exclude from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame containing the IAA metrics table and the imbalance metrics
    """

    if engine == "argilla":
        raise DeprecationWarning("Argilla engine is deprecated")
    elif engine == "nltk":
        calculator = nltk_iaa
    else:
        raise NotImplementedError(f"Unknown engine {engine}")

    assert isinstance(calculator, Callable)

    rows = []
    for question in questions:
        rows.append(calculator(dataset, question, rejected_users, exclude_dont_know))

    return pd.DataFrame(rows)


def nltk_iaa(
    dataset: FeedbackDataset,
    question: str,
    rejected_users: Optional[list[str]],
    exclude_dont_know: bool = False,
) -> dict:
    """Calculates the inter-annotator agreement for a single question in a dataset."""
    df = transform_to_df(dataset)
    df = format_df(df, question)

    likert_question = question in likert_questions

    triplets = [
        (
            str(x.user),
            str(x.q_id),
            float(x.response),
        )
        for _, x in df.iterrows()
    ]

    if rejected_users is not None:
        LOGGER.info(f"Excluding users: {rejected_users}")
        triplets = [
            (user, q_id, response)
            for user, q_id, response in triplets
            if user not in rejected_users
        ]

    if exclude_dont_know:
        LOGGER.info(f"Excluding responses: {default_reject_responses}")
        triplets = [
            (user, q_id, response)
            for user, q_id, response in triplets
            if response not in default_reject_responses
        ]

    return {
        "count": len(triplets),
        "krippendorff_alpha": krippendorff_alpha(triplets, likert_question),
        "average_pairwise_agreement": pairwise_metrics(
            triplets, "agreement", likert_question
        ),
        "average_pairwise_cohen_kappa": pairwise_metrics(
            triplets, "cohen", likert_question
        ),
        "question": question,
    } | imbalance_measurements(triplets)


def abs_interval_distance(a: int | float, b: int | float, scale: int = 5) -> float:
    """Calculates the absolute interval distance"""
    return abs(a - b) / scale


def krippendorff_alpha(
    triples: list[tuple[str, str, float]], likert_question: bool
) -> float:
    """Calculates the Krippendorf's alpha for a list of triples"""
    distance = binary_distance if not likert_question else abs_interval_distance
    task = AnnotationTask(data=triples, distance=distance)
    return task.alpha()


def pairwise_metrics(
    triples: list[tuple[str, str, float]],
    metric: Literal["cohen", "agreement"],
    likert_question: bool = False,
) -> Optional[float]:
    """
    Calculates the pairwise metrics for a list of triples

    Args:
        triples (list[tuple[str, str, str | float]]): a list of triples (user, question, response)
        metric (Literal["cohen", "agreement"]): the metric to calculate
        likert_question (bool, optional): whether the question is a likert scale question. Defaults to False.

    Returns:
        float: the average pairwise metric
    """
    df = pd.DataFrame(triples, columns=["user", "question", "response"])
    coders = df.user.unique().tolist()

    if metric == "cohen" and likert_question:
        LOGGER.warning("Cohen's Kappa is not suitable for Likert scale questions")
        return None

    agreements = []

    for c1, c2 in combinations(coders, 2):
        _df = df[(df["user"] == c1) | (df["user"] == c2)]
        _df = _df.drop_duplicates(
            subset=["question", "user", "response"]
        )  # needed for duplicated records
        valid_questions = _df[_df["question"].duplicated()]["question"].unique()
        _df = _df[_df["question"].isin(valid_questions)]

        if _df.shape[0] > 0:
            if metric == "agreement":
                agreements.append(
                    _pairswise_agreement(_df, likert_question=likert_question)
                )
            elif metric == "cohen":
                agreements.append(_pairwise_cohen(_df, c1, c2))
            else:
                raise NotImplementedError(f"Unknown metric {metric}")

    return np.mean(agreements).astype(float)


def _pairswise_agreement(df: pd.DataFrame, likert_question: bool = False) -> float:
    """Calculates the pairwise agreement ratio"""
    if likert_question:

        def agg_func(x):  # pyright: ignore
            return reduce(abs_interval_distance, x)
    else:

        def agg_func(x):  # pyright: ignore
            return x.nunique() == 1

    return df.groupby("question").agg({"response": agg_func}).response.mean()


def _pairwise_cohen(df: pd.DataFrame, c1: str, c2: str) -> float:
    """
    Calculates the Cohen Kappa score for the two coders

    This function is reimplemented using the pairwise scores, because the NLTK implementation
    does not support cases where the coders have not annotated the same questions. Hence the
    dataset is sliced by pairs, and the micro Cohen Kappa is calculated for each pair, which
    is then averaged.
    """
    ann1, ann2 = [], []
    for q in df["question"].unique():
        _q = df[df["question"] == q]
        ann1.append(str(_q[_q["user"] == c1]["response"].values[0]))
        ann2.append(str(_q[_q["user"] == c2]["response"].values[0]))

    if len(set(ann1 + ann2)) > 1:
        return cohen_kappa_score(ann1, ann2)
    else:
        return 1.0


def imbalance_measurements(triples: list[tuple[str, str, float]]) -> dict:
    """
    Measures the imbalance of the annotations

    Ouputs 3 different metrics for the imbalance of the annotations:
    - skew: the skewness of the distribution. 0 is symmetric, negative is left-skewed, positive is right-skewed.
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    - imbalance_ratio: the ratio of the most frequent annotation to the least frequent annotation
        (the lower the better, 1 is perfect balance).
    - entropy: the entropy of the distribution. The higher the entropy, the more balanced the distribution (c.f.
        less sum surprisal means the annotations are skewed towards a single value NB this isn't generally true!)
        See more in: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    observations = [x[2] for x in triples]
    probs = [v / len(observations) for v in Counter(observations).values()]

    return {
        "skew": skew(observations),
        "imbalance_ratio": max(probs) / min(probs),
        "entropy": -sum([p * math.log(p) for p in probs]),
    }


def argilla_iaa(
    dataset: rg.FeedbackDataset,
    question: str,
    rejected_users: Optional[list[str]],
    exclude_dont_know: bool = False,
) -> dict:
    """Calculates the inter-annotator agreement for a single question in a dataset."""
    _dataset = deepcopy(dataset)

    if rejected_users is not None:
        _dataset = _filter_dataset_by_users(_dataset, rejected_users)

    if exclude_dont_know:
        _dataset = _filter_dataset_by_response(
            _dataset, question, default_reject_responses
        )

    metric = AgreementMetric(
        dataset=_dataset,
        question_name=question,
    )
    metrics_report = metric.compute("alpha")
    return metrics_report.__dict__ | {"question": question}
