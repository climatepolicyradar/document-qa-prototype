from src.evaluation.evaluator import Score
from src.evaluation.faithfulness.g_eval_faithfulness import GEvalFaithfulness
from tests.evaluation.util import e2e_data

from unittest.mock import MagicMock, patch

assert e2e_data  # ensure that pyright doesn't remove this


def test_g_eval(e2e_data):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="3")
    patch(
        "src.evaluation.g_eval.get_llm",
        return_value=mock_llm,
    ).start()

    evaluator = GEvalFaithfulness()

    for e2e_gen in e2e_data:
        result = evaluator.evaluate(e2e_gen)
        assert isinstance(result, Score)
        assert result.score == 0.5
