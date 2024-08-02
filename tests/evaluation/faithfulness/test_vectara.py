import torch

from src.evaluation.evaluator import Score
from src.evaluation.faithfulness.vectara import Vectara
from tests.evaluation.util import e2e_data

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DebertaV2ForSequenceClassification
from unittest.mock import patch

assert DebertaV2ForSequenceClassification  # ensure that pyright doesn't remove this
assert e2e_data  # ensure that pyright doesn't remove this


def test_vectara(e2e_data):
    evaluator = Vectara()

    patch(
        "tests.evaluation.faithfulness.test_vectara.DebertaV2ForSequenceClassification.__call__",
        return_value=SequenceClassifierOutput(
            loss=None,
            logits=torch.tensor([[-2.4105]]),
            hidden_states=None,
            attentions=None,
        ),
    ).start()

    for e2e_gen in e2e_data:
        result = evaluator.evaluate(e2e_gen)
        assert isinstance(result, Score)
