import pytest
from src.models.data_models import EndToEndGeneration, RAGRequest, RAGResponse


@pytest.fixture
def sample_end_to_end_generation():
    rag_request = RAGRequest(
        query="Test query",
        document_id="test_doc_id",
        top_k=5,
        model="test_model",
        prompt_template="test_template",
    )
    rag_response = RAGResponse(
        text="#COT#Inner monologue#/COT#Actual answer",
        retrieved_documents=[],
        query="Test query",
    )
    return EndToEndGeneration(
        config={}, rag_request=rag_request, rag_response=rag_response
    )


def test_get_answer_with_cot_removal(sample_end_to_end_generation):
    answer = sample_end_to_end_generation.get_answer(remove_cot=True)
    assert answer == "Actual answer"


def test_get_answer_without_cot_removal(sample_end_to_end_generation):
    answer = sample_end_to_end_generation.get_answer(remove_cot=False)
    assert answer == "#COT#Inner monologue#/COT#Actual answer"


def test_get_answer_no_cot_tags(sample_end_to_end_generation):
    sample_end_to_end_generation.rag_response.text = "Simple answer without COT tags"
    answer = sample_end_to_end_generation.get_answer(remove_cot=True)
    assert answer == "Simple answer without COT tags"


def test_get_answer_no_rag_response():
    generation = EndToEndGeneration(
        config={},
        rag_request=RAGRequest(query="Test", document_id="test_doc"),
        rag_response=None,
    )
    answer = generation.get_answer()
    assert answer == ""
