import pytest

from src.controllers.VespaController import VespaController


@pytest.fixture
def vespa_controller():
    return VespaController()


def test_query(vespa_controller: VespaController):
    query = "What are the key climate policy contributions?"
    doc_id = "UNFCCC.party.859.0"

    response = vespa_controller.query(query, doc_id)

    assert response is not None
    assert "root" in response
    assert "children" in response["root"]
    assert len(response["root"]["children"]) > 0
