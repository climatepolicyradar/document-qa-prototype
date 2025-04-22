import pytest
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.online.inference import get_llm
from langchain_core.documents import Document as LangChainDocument


llm = get_llm("vertexai", "llama3-8b-chat")


@pytest.fixture
def mock_retriever():
    return [
        LangChainDocument(
            document_id="1",
            page_content="This is the first document",
            metadata={"title": "First Document"},
        ),
        LangChainDocument(
            document_id="2",
            page_content="This is the second document",
            metadata={"title": "Second Document"},
        ),
    ]

"""
THESE TESTS FAIL UNTIL WE FIX THE VERTEX AI MODEL

def test_vertex_ai_model():
    response = llm.generate(prompts=["What is the capital of the moon?"])
    assert response is not None


def test_all_models_e2e():
    sc = ScenarioController.from_config("test_vertex_models")
    rc = RagController(observe=False)

    for scenario in sc.scenarios:
        print(f"Testing scenario: {scenario}")
        result = rc.run_llm(scenario, {})
        assert len(str(result)) > 0
        assert "fin." in str(result).lower()
"""