import pytest
from unittest.mock import patch, MagicMock
from src.controllers.DocumentController import DocumentController
from cpr_data_access.models import BaseDocument
import pandas as pd
from pydantic import AnyHttpUrl

@pytest.fixture
def document_controller():
    return DocumentController()

def test_load_metadata(document_controller):
    assert not document_controller.metadata_df.empty, "📄 Metadata DataFrame should not be empty"

def test_create_base_document_success(document_controller):
    document = document_controller.create_base_document('CCLW.executive.9494.3809')
    print(document.document_source_url)
    assert isinstance(document, BaseDocument), "📄 Should return a BaseDocument instance"
    assert document.document_id == 'CCLW.executive.9494.3809', "📄 Document ID should match"
    assert document.document_name == 'National energy -climate plan: Part A - National Plan', "📄 Document name should match"
    assert document.document_source_url == AnyHttpUrl('https://app.climatepolicyradar.org/document/national-energy-climate-plan-part-a-national-plan_2e62'), "📄 Document URL should match"

def test_create_base_document_not_found(document_controller):
    with pytest.raises(AssertionError, match="❌ Document ID not found in metadata"):
        document_controller.create_base_document('nonexistent_doc')