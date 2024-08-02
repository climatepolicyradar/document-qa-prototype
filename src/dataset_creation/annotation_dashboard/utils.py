import argilla as rg
import pandas as pd

from argilla import FeedbackDataset
from typing import Optional
from argilla.client.feedback.schemas.remote.records import ResponseSchema
from src.logger import get_logger

LOGGER = get_logger(__name__)

map_to_binary = {
    "YES": 1,
    "NO": 0,
    "NO_WITH_CONTEXT": 0.5,
    "PARTIAL": 0.5,
    "0": 0 / 5,
    "1": 1 / 5,
    "2": 2 / 5,
    "3": 3 / 5,
    "4": 4 / 5,
    "5": 5 / 5,
}
default_reject_responses = ["DONT_KNOW", "NOT_APPLICABLE"]

likert_questions = ["overall-quality"]


def transform_to_df(dataset: rg.FeedbackDataset) -> pd.DataFrame:
    """
    Transforms the dataset to a pandas DataFrame for easier handling.

    This dataset only contains the fields relevant for disagreement:
    - q_id: the question id
    - user: user
    - response: response value
    """
    _response_data = []

    for i in dataset:
        for r in i.responses:
            _response_data.append(
                {"q_id": i.metadata["q_id"], "response": r, "user": str(r.user_id)}
            )

    return pd.DataFrame(_response_data)


def format_df(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Formats the dataset by mapping responses to more managable formats

    NOTE: this function will drop rows with missing values.
    """
    df["response"] = df["response"].apply(lambda i: _get_response_value(i, question))
    original_size = df.shape[0]
    df.dropna(subset=["response"], inplace=True)
    LOGGER.info(
        f"Filtered dataset by removing {original_size - df.shape[0]} rows with missing values"
    )
    return df


def _get_response_value(response: ResponseSchema, question: str) -> Optional[int]:
    """Gets the response value for a question."""
    _question_value = response.values.get(question)
    _value = getattr(_question_value, "value", None)
    if _value is None:
        return None
    return map_to_binary.get(_value, None)


def _filter_dataset_by_response(
    dataset: FeedbackDataset, q: str, rejected_responses: list[str]
) -> FeedbackDataset:
    old, new = 0, 0

    for row in dataset.records:
        old += len(row.responses)
        _resp = [
            _r
            for _r in row.responses
            if getattr(_r.values.get(q), "value", None) not in rejected_responses
        ]
        row.responses = _resp
        new += len(_resp)

    LOGGER.debug(
        f"Filtered {old - new} responses out of {old} total responses for question {q}"
    )

    return dataset


def _filter_dataset_by_users(
    dataset: FeedbackDataset, rejected_users: list[str]
) -> rg.FeedbackDataset:
    old, new = 0, 0

    for row in dataset.records:
        old += len(row.responses)
        _resp = [_r for _r in row.responses if str(_r.user_id) not in rejected_users]
        row.responses = _resp
        new += len(_resp)

    LOGGER.debug(f"Filtered {old - new} responses out of {old} total responses.")

    return dataset
