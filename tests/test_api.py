import unittest.mock as mock
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_get_health():
    with mock.patch("wandb.init") as mock_wandb:
        mock_wandb.return_value = None  # or whatever value you want to return
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}
