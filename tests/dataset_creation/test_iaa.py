import json

from argilla.client.feedback.schemas import LabelQuestion

from datasets import Dataset
from unittest.mock import MagicMock, patch

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


@patch(
    "src.dataset_creation.annotation_dashboard.iaa.FeedbackDataset.question_by_name",
    return_value=LabelQuestion,
)
def test_calculate_iaa_table(self):
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

    # df_argilla = calculate_iaa_table(
    #     dataset, ["q1"], engine="argilla", exclude_dont_know=False, rejected_users=None
    # )

    # assert df_argilla.shape == (1, 4)

    # df_nltk = calculate_iaa_table(
    #     dataset, ["q1"], engine="nltk", exclude_dont_know=False, rejected_users=None
    # )

    # assert (df_nltk["result"] == df_argilla["result"]).all()
