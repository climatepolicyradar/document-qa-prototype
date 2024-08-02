import pytest
import yaml

from src.controllers.RagController import RagController
from src.online.inference import LLMTypes
from src.dataset_creation.query_utils import load_template_map

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
    assert doc_text['root']['fields']['totalCount'] == len(doc_text['root']['children'])

def test_get_available_documents(rag_controller):
    docs = rag_controller.get_available_documents()
    assert docs is not None
    assert len(docs) > 0
    assert docs['root']['children'][0]['id'] is not None
    
    
def test_generate_queries(rag_controller):
    doc_id = "UNFCCC.party.859.0"
    doc = rag_controller.get_document_text(doc_id)
    
    queries = rag_controller.generate_queries(
        model="gpt-4", 
        generation_engine=LLMTypes.OPENAI, 
        prompt_template="", 
        document=doc
    )
    assert queries is not None
    assert len(queries) > 0