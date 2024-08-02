from collections import Counter
import argilla as rg

import math
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib.figure import Figure
from itertools import combinations

from src.dataset_creation.annotation_dashboard.utils import (
    _filter_dataset_by_users,
    transform_to_df,
    format_df,
    _filter_dataset_by_response,
    default_reject_responses,
)
from src.logger import get_logger


LOGGER = get_logger(__name__)


def disagreement_plot(
    dataset: rg.FeedbackDataset,
    question: str,
    id_to_user: dict[str, str],
    rejected_users: Optional[list[str]] = None,
    exclude_dont_know: bool = False,
    user: Optional[str] = None,
) -> tuple[Figure, Optional[Figure]]:
    """
    Creates the disagreement chart, showing the number of disagreements between users.

    TODO: handling the likert scale questions.

    Args:
        dataset (rg.FeedbackDataset): The dataset to plot
        question (str): The question to filter the dataset
        id_to_user (dict[str, str]): A dictionary mapping user ids to usernames
        rejected_users (Optional[list[str]], optional): A list of users to exclude from the dataset. Defaults to None.
        exclude_dont_know (bool, optional): Whether to exclude the "DONT_KNOW" / "NOT_APPLICABLE" responses. Defaults to False.
        user (Optional[str], optional): The user to filter the dataset. Defaults to None.

    Returns:
        tuple[Figure, Optional[Figure]]: The pyplot figure of a set of pie charts showing the ratio of agreements and disagreements
             and optionally a bar chart showing the disagreement ratios of the annotators.
    """
    if exclude_dont_know:
        dataset = _filter_dataset_by_response(
            dataset, question, default_reject_responses
        )

    if rejected_users is not None:
        dataset = _filter_dataset_by_users(dataset, rejected_users)

    df = transform_to_df(dataset)
    df = format_df(df, question)
    df["user"] = df["user"].apply(id_to_user.get)

    LOGGER.info(
        f"Plotting disagreement chart for question: {question} with {df.shape[0]} responses"
    )

    disagreements, agreements = get_pairwise_disagreements_and_agreements(df)

    if user:
        disagreements = [i for i in disagreements if user in i]
        agreements = [i for i in agreements if user in i]

    disagreement_counts = Counter(disagreements)
    agreement_counts = Counter(agreements)

    all_pairs = set(disagreements) | set(agreements)

    num_cols = 5
    num_rows = math.ceil(len(all_pairs) / num_cols)

    assert len(all_pairs) <= num_rows * num_cols

    fig_pairwise_disagreements, axs_pairwise_disagreements = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
    )

    for i, pair in enumerate(
        sorted(
            all_pairs,
            key=lambda x: disagreement_counts[x] + agreement_counts[x],
            reverse=True,
        )
    ):
        if num_rows == 1 or num_cols == 1:
            _ax = axs_pairwise_disagreements[i]
        else:
            _ax = axs_pairwise_disagreements[i // num_cols, i % num_cols]
        _ax.set_title(
            f"{format_pair_name(pair)}: {disagreement_counts[pair] + agreement_counts[pair]}"
        )
        _ax.pie(
            [disagreement_counts[pair], agreement_counts[pair]],
            labels=["Disagree", "Agree"],
            colors=["red", "green"],
            autopct="%1.1f%%",
            startangle=90,
        )

    for i in range(len(all_pairs), num_rows * num_cols):
        if num_rows == 1 or num_cols == 1:
            _ax = axs_pairwise_disagreements[i]
        else:
            _ax = axs_pairwise_disagreements[i // num_cols, i % num_cols]
        _ax.axis("off")

    plt.suptitle(f"Question: {question}", fontsize=20)
    plt.tight_layout()

    fig_disagreement_ratios = None
    if not user:
        users = df["user"].unique().tolist()

        values = []
        for u in users:
            _d = n_user_in_pairs(u, disagreements)
            _a = n_user_in_pairs(u, agreements)
            if _d + _a == 0:
                values.append(0)
            else:
                values.append(_d / (_d + _a))

        fig_disagreement_ratios, ax_disagreement_ratios = plt.subplots()
        ax_disagreement_ratios.bar(users, values)
        ax_disagreement_ratios.set_title("Disagreement Ratio")
        plt.xticks(rotation=90)
        plt.tight_layout()

    return fig_pairwise_disagreements, fig_disagreement_ratios


def n_user_in_pairs(user: str, pairs: list[str]) -> int:
    return sum(user in i for i in pairs)


def get_pairwise_disagreements_and_agreements(
    df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Using the dataframe, returns a list of disagreements and agreements between users.

    Importantly the agreement is taken in a pairwise manner, i.e. for questions with more than 2 responses,
    all combinations of users are considered.

    Args:
        df (pd.DataFrame): The dataframe to use with columns "q_id", "user", and "response"
        question (str): The question to filter the dataset

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of disagreements and agreements. The first tuple
            contains the disagreements, the second contains the agreements. Each element in the list is a string
            with the two users separated by a dash.
    """
    disagreeing_users = []
    agreeing_users = []

    for _, question_group in df.groupby("q_id"):
        if len(question_group) > 1:
            group_disagreements, group_agreements = _pairwise_agreements(question_group)
            disagreeing_users.extend(group_disagreements)
            agreeing_users.extend(group_agreements)

    return disagreeing_users, agreeing_users


def _pairwise_agreements(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Checks the pairwise agreements of a DataFrame for a single question."""
    agreements, disagreements = [], []
    user_pairs = combinations(df["user"].unique(), 2)

    for pair in user_pairs:
        _pair_string = "-".join(sorted(pair))
        if df[df["user"].isin(pair)]["response"].nunique() == 1:
            agreements.append(_pair_string)
        else:
            disagreements.append(_pair_string)

    return disagreements, agreements


def format_pair_name(users_str: str) -> str:
    users_str = users_str.replace("-", " & ").replace("_", " ")
    return " ".join(i.capitalize() for i in users_str.split())
