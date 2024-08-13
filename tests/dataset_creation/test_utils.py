from src.models.data_models import RAGRequest
import yaml
import pytest

from src.controllers.ScenarioController import ScenarioController


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
        - prompt_template: response/FAITHFULQA_SCHIMANSKI_CITATION_QA_TEMPLATE_MODIFIED
        - prompt_template: response/adversarial
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
    sc = ScenarioController.from_config_dict(config)

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

    assert len(sc.scenarios) == 2
    assert all(a.top_k_retrieval_results is not None for a in sc.scenarios)
    assert all(
        a.generation_engine in {"openai", "huggingface", "gemini"} for a in sc.scenarios
    )

    config["model_selection"] = "all"

    sc = ScenarioController.from_config_dict(config)

    assert len(sc.scenarios) == 12
    assert all(a.top_k_retrieval_results is not None for a in sc.scenarios)

    config["template_selection"] = "random"

    sc = ScenarioController.from_config_dict(config)

    assert len(sc.scenarios) == 6
    assert all(a.top_k_retrieval_results is not None for a in sc.scenarios)
