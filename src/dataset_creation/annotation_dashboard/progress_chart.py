import matplotlib.pyplot as plt
import argilla as rg

from collections import Counter
from matplotlib.figure import Figure


def response_count_plot(
    dataset: rg.FeedbackDataset, id_to_user: dict[str, str]
) -> Figure:
    """
    Plots the number of responses per user.

    Args:
        dataset (rg.FeedbackDataset): The merged dataset to plot
        id_to_user (dict[str, str]): A dictionary mapping user ids to usernames

    Returns:
        Figure: The pyplot figure with horizontal bar plot
    """
    _users = list(
        id_to_user[str(_resp.user_id)] for r in dataset.records for _resp in r.responses
    )

    response_counts = Counter(_users)

    fig, ax = plt.subplots(figsize=(10, 10))

    _bar_color = "skyblue"
    sorted_vals = sorted(
        response_counts.items(), key=lambda item: item[1], reverse=False
    )

    names = [name_parser(i[0]) for i in sorted_vals]
    values = [i[1] for i in sorted_vals]

    plt.barh(names, values, color=_bar_color)

    plt.yticks(fontsize=10)

    plt.xlim(0, 260)
    plt.ylim(-0.5, len(names) - 0.5)

    for i, v in enumerate(values):
        if v:
            if v > 4:
                _v_offset = v / 2 - 1
                color = "white"
            else:
                _v_offset = v + 1
                color = _bar_color
            ax.text(_v_offset, i - 0.2, str(v), color=color, fontweight="bold")

    plt.tight_layout()
    return fig


def name_parser(user: str) -> str:
    return " ".join(i.capitalize() for i in user.split("_"))
