import pytest
from unittest.mock import patch, Mock
from src.evaluation.evaluator import Score
from src.evaluation.faithfulness.vectara import Vectara
from src.models.data_models import EndToEndGeneration, RAGResponse
import uuid

@pytest.fixture
def mock_requests_post():
    with patch('requests.Session.post') as mock_post:
        yield mock_post

@pytest.fixture
def mock_e2e_generation():
    return EndToEndGeneration(
        uuid=str(uuid.uuid4()),
        rag_response=RAGResponse(
            retrieved_passages=["This is a test passage."]
        ),
        answer="This is a test answer."
    )

def test_vectara_successful_api_call(mock_e2e_generation, mock_requests_post):
    # Setup
    evaluator = Vectara()
    mock_response = Mock()
    mock_response.json.return_value = {"score": 0.75}
    mock_response.raise_for_status.return_value = None
    mock_requests_post.return_value = mock_response

    # Test
    result = evaluator.evaluate(mock_e2e_generation)
    
    # Assertions
    assert isinstance(result, Score)
    assert result.score == 0.75
    assert result.type == "faithfulness"
    assert result.name == "vectara"
    assert result.gen_uuid == mock_e2e_generation.uuid

    # Verify API call
    mock_requests_post.assert_called_with(
        "https://vectara-api.labs.climatepolicyradar.org/evaluate",
        json={
            "context": mock_e2e_generation.rag_response.retrieved_passages_as_string(),
            "response": mock_e2e_generation.get_answer()
        }
    )

def test_vectara_api_error(mock_e2e_generation, mock_requests_post):
    # Setup
    evaluator = Vectara()
    mock_requests_post.side_effect = Exception("API Error")

    # Test
    result = evaluator.evaluate(mock_e2e_generation)
    
    # Assertions
    assert result is None

def test_vectara_no_rag_response(mock_e2e_generation):
    # Setup
    evaluator = Vectara()
    mock_e2e_generation.rag_response = None

    # Test
    result = evaluator.evaluate(mock_e2e_generation)

    # Assertions
    assert result is None
