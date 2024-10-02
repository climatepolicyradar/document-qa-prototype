import pytest
import yaml

from src.controllers.RagController import RagController


@pytest.fixture
def rag_controller():
    return RagController()


@pytest.fixture()
def config():
    return yaml.safe_load(
        """
        models:
        - generation_engine: openai
          model: gpt-4
        model_selection: random
        seed_queries_path: s3://project-rag/data/queries/seed_samples.jsonl
        prompt_templates:
        - prompt_template: "query_from_product_queries.txt"
        template_selection: all
        """
    )


def test_get_document_text(rag_controller):
    doc_id = "UNFCCC.party.859.0"
    doc_text = rag_controller.get_document_text(doc_id)
    print(f"Found {len(doc_text['root']['children'])} text blocks")
    assert doc_text is not None
    assert doc_text["root"]["fields"]["totalCount"] == len(doc_text["root"]["children"])
