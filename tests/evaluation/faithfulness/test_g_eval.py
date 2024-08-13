import uuid
import pytest
from src.evaluation.evaluator import Score
from src.evaluation.faithfulness.g_eval_faithfulness import GEvalFaithfulness

from unittest.mock import MagicMock

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


def test_g_eval(mock_e2e_generation):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="3")

    evaluator = GEvalFaithfulness()
    evaluator.model = mock_llm

    result = evaluator.evaluate(mock_e2e_generation)
    assert isinstance(result, Score)
    assert result.score == 0.5
