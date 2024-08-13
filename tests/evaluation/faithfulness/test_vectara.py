import pytest
from unittest.mock import patch, Mock
from src.evaluation.evaluator import Score
from src.evaluation.faithfulness.vectara import Vectara
from tests.evaluation.util import e2e_data

@pytest.fixture
def mock_requests_post():
    with patch('requests.Session.post') as mock_post:
        yield mock_post

def test_vectara_successful_api_call(e2e_data, mock_requests_post):
    # Setup
    evaluator = Vectara()
    mock_response = Mock()
    mock_response.json.return_value = {"score": 0.75}
    mock_response.raise_for_status.return_value = None
    mock_requests_post.return_value = mock_response

    # Test
    for e2e_gen in e2e_data:
        result = evaluator.evaluate(e2e_gen)
        
        # Assertions
        assert isinstance(result, Score)
        assert result.score == 0.75
        assert result.type == "faithfulness"
        assert result.name == "vectara"
        assert result.gen_uuid == e2e_gen.uuid

    # Verify API call
    mock_requests_post.assert_called_with(
        "https://vectara-api.labs.climatepolicyradar.org/evaluate",
        json={
            "context": e2e_gen.rag_response.retrieved_passages_as_string(),
            "response": e2e_gen.get_answer()
        }
    )

def test_vectara_api_error(e2e_data, mock_requests_post):
    # Setup
    evaluator = Vectara()
    mock_requests_post.side_effect = Exception("API Error")

    # Test
    for e2e_gen in e2e_data:
        result = evaluator.evaluate(e2e_gen)
        
        # Assertions
        assert result is None

def test_vectara_no_rag_response(e2e_data):
    # Setup
    evaluator = Vectara()
    e2e_gen_no_rag = e2e_data[0]
    e2e_gen_no_rag.rag_response = None

    # Test
    result = evaluator.evaluate(e2e_gen_no_rag)

    # Assertions
    assert result is None
