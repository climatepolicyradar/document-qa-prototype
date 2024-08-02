from src.dataset_creation.utils import create_args
from src.models.data_models import RAGRequest
import yaml
import pytest

from dataclasses import asdict

from src.dataset_creation.utils import create_args
from src.models.data_models import RAGRequest


@pytest.fixture()
def config():
    return yaml.safe_load(
        """
        models:
        - generation_engine: openai
          model: gpt-3.5-turbo
        - generation_engine: openai
          model: gpt-4
        - generation_engine: huggingface
          model: mistralai/Mistral-7B-Instruct-v0.2
        - generation_engine: huggingface
          model: mistralai/Mixtral-8x7B-Instruct-v0.1
        - generation_engine: gemini
          model: gemini-pro
        - generation_engine: gemini
          model: gemini-1.0-pro-001
        model_selection: random
        queries_path: data/queries/queries.json
        prompt_templates:
        - prompt_template: FAITHFULQA_SCHIMANSKI_CITATION_QA_TEMPLATE_MODIFIED
        - prompt_template: adversarial
        template_selection: all
        retrieval:
          - retrieval_window: 0
            top_k: 10
          - retrieval_window: 1
            top_k: 3
        retrieval_selection: random
        """
    )


def test_create_args(config):
    args = create_args(config)
    required_config_keys = {
        "generation_engine",
        "model",
        "prompt_template",
        "retrieval_window",
        "top_k",
    }

    # All keys in the config file must match up to properties in the RAGRequest model
    assert all(
        _key in RAGRequest.model_json_schema()["properties"].keys()
        for _key in required_config_keys
    )

    assert len(args) == 2
    assert all(asdict(a).keys() == required_config_keys for a in args)
    assert all(a.top_k is not None for a in args)
    assert all(a.generation_engine in {"openai", "huggingface", "gemini"} for a in args)

    config["model_selection"] = "all"

    args = create_args(config)

    assert len(args) == 12
    assert all(asdict(a).keys() == required_config_keys for a in args)
    assert all(a.top_k is not None for a in args)

    config["template_selection"] = "random"

    args = create_args(config)

    assert len(args) == 6
    assert all(asdict(a).keys() == required_config_keys for a in args)
    assert all(a.top_k is not None for a in args)
