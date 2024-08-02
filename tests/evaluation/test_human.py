import json
import pytest

from argilla.client.feedback.schemas import LabelQuestion

from datasets import Dataset
from unittest.mock import MagicMock, patch
from src.evaluation.human import argilla_to_scores, AggregationStrategy


@pytest.fixture
def mock_dataset():
    responses = [
        ["YES", "NO"],
        ["PARTIAL", "YES", "NO"],
        ["YES", "YES", "NO"],
        ["YES"],
        ["NO"],
        ["DONT_KNOW"],
    ]

    data = {
        "q1": [
            [{"user_id": f"user{i}", "value": r} for i, r in enumerate(rs)]
            for rs in responses
        ],
        "metadata": [json.dumps({"q_id": f"qid{i}"}) for i in range(len(responses))],
        "text": [f"text{i}" for i in range(len(responses))],
    }

    dataset = MagicMock()

    responses = []
    for i, m in zip(data["q1"], data["metadata"]):
        responses.append(
            MagicMock(
                responses=[
                    MagicMock(
                        user_id=r["user_id"],
                        values={"q1": MagicMock(value=r["value"])},
                    )
                    for r in i
                ],
                metadata=json.loads(m),
            )
        )

    dataset.__iter__.return_value = responses
    dataset.question_by_name.return_value = LabelQuestion(
        name="q1", labels=["YES", "NO"]
    )

    dataset.filter_by.return_value = dataset
    dataset.format_as.return_value = Dataset.from_dict(data)
    return dataset


@patch(
    "src.evaluation.human.FeedbackDataset.question_by_name",
    return_value=LabelQuestion,
)
@pytest.mark.parametrize(
    ("strategy", "results"),
    (
        (AggregationStrategy.AVERAGE, [0.5, 0.5, 0.6666666666666666, 1.0, 0.0]),
        (AggregationStrategy.COMPLETE, [0.0, 0.0, 0.0, 1.0, 0.0]),
        (
            AggregationStrategy.MAJORITY,
            [
                1.0,
                1.0,
                0.0,
            ],  # NOTE: here only 3 Scores, because ["YES", "NO"] and ["PARTIAL", "YES", "NO"] returns None
        ),
    ),
)
def test_argilla_to_scores(
    self, strategy: AggregationStrategy, results: list[float], mock_dataset
):
    scores = argilla_to_scores(mock_dataset, strategy, ["q1"])

    assert len(scores) == len(results)
    assert [score.score for score in scores] == results
