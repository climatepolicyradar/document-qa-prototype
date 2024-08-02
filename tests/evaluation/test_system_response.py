from src.evaluation.evaluator import Score
from src.evaluation.system_response import SystemResponse
from tests.evaluation.util import e2e_data

import pytest

assert e2e_data  # ensure that pyright doesn't remove this


@pytest.fixture
def expected_system_responds_results():
    return [1, 0, 1, 0.5, 0, 0]


def test_system_response_eval(e2e_data, expected_system_responds_results):
    evaluator = SystemResponse()

    for idx, e2e_gen in enumerate(e2e_data):
        result = evaluator.evaluate(e2e_gen)
        assert isinstance(result, Score)
        assert result.type == "system_response"
        assert result.name == "substring_match"
        assert result.score == expected_system_responds_results[idx]
