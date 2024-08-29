import uuid
from src.models.data_models import Score
from src.evaluation.system_response.system_response import SystemResponse

import pytest

from src.models.data_models import EndToEndGeneration, RAGRequest, RAGResponse


@pytest.fixture
def mock_e2e_generation():
    return EndToEndGeneration(
        config={},
        uuid=str(uuid.uuid4()),
        rag_request=RAGRequest(query="This is a test query.", document_id="test_id"),
        rag_response=RAGResponse(
            text="This is a test answer.",
            query="This is a test query.",
            retrieved_documents=[{"page_content": "This is a test passage."}],
        ),
    )


@pytest.fixture
def mock_e2e_no_response():
    return EndToEndGeneration(
        config={},
        uuid=str(uuid.uuid4()),
        rag_request=RAGRequest(query="This is a test query.", document_id="test_id"),
        rag_response=RAGResponse(
            text="I cannot provide an answer to this question.",
            query="This is a test query.",
            retrieved_documents=[{"page_content": "This is a test passage."}],
        ),
    )


def test_system_response_eval(mock_e2e_generation, mock_e2e_no_response):
    evaluator = SystemResponse()

    result = evaluator.evaluate(mock_e2e_generation)
    assert isinstance(result, Score)
    assert result.type == "system_response"
    assert result.name == "substring_match"
    assert result.score == 1.0

    result = evaluator.evaluate(mock_e2e_no_response)
    assert result.score == 0.0
