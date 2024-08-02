import pytest
from src.online.inference import get_llm
from langchain_core.documents import Document as LangChainDocument

from src.prompts.template_building import get_citation_template
from src.online.pipeline import rag_chain
from test_pipeline import MockVectorStoreRetriever

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


def test_vertex_ai_model():
    response = llm.generate(prompts=["What is the capital of the moon?"])
    print(response)
    assert response is not None


def test_rag_chain_with_vertex(mocker, mock_retriever):
    retriever = MockVectorStoreRetriever(mock_retriever)

    mocker.patch("src.online.pipeline.text_is_too_long_for_model", return_value=False)

    _chain = rag_chain(
        citation_template=get_citation_template(
            "FAITHFULQA_SCHIMANSKI_CITATION_QA_TEMPLATE_MODIFIED"
        ),
        llm=llm,
        retriever=retriever,  # type: ignore
        window_radius=0,
    )

    response = _chain.invoke("This is the query")

    assert mock_retriever[0].page_content in response["answer"]
    assert len(response["documents"]) == 2
    assert response["query_str"] == "This is the query"
    assert [w[0] for w in response["windows"]] == response["documents"]
    assert "[0] This is the first document" in response["context_str"]
    assert "[1] This is the second document" in response["context_str"]
    assert response["llm_output"] == response["answer"]
