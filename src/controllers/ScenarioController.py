# type: ignore
from typing import Optional
import jinja2
from pathlib import Path

import s3fs
import yaml
import src.config as config
import json
import random

from src.prompts.template_building import jinja_template_loader
from src.logger import get_logger
from src.models.data_models import (
    Prompt,
    Query,
    Scenario,
)
from enum import Enum
from src.flows.utils import get_secret

LOGGER = get_logger(__name__)
random.seed(42)

"""
ScenarioController: to manage config/scenarios
- [x] Load scenario from config file
- [x] get seed queries
- [x] set seed queries
- [x] get template map
- [x] set template map 
- [x] create args ?? == scenario! But with a document associated as well. 
"""


class SelectionStrategy(Enum):
    """Represents a strategy for selecting models, prompts, retrieval, etc for a scenario"""

    RANDOM = "random"
    ALL = "all"


"""
Refactor list:
generate_responses: df: pd.DataFrame,referenced create_generations.py
"""


class ScenarioController:
    """Controller for scenarios, a representation of the configuration of a pipeline run"""

    scenarios: list[Scenario]
    config: dict

    def __init__(self, scenarios: Optional[list[Scenario]] = None):
        self.scenarios = scenarios if scenarios is not None else []

    def get_config(self) -> dict:
        """Returns the config dictionary"""
        return self.config

    @classmethod
    def from_config(cls, config_path: str) -> "ScenarioController":  # type: ignore
        """Load a set of scenarios from a config file"""
        if "yaml" not in config_path:
            config_path = config_path + ".yaml"
        if "/" not in config_path:
            config_path = f"src/configs/{config_path}"

        LOGGER.info(f"🤖 Loading scenarios from config: {config_path}")
        config_path_p = Path(config_path)
        cls.config = yaml.safe_load(config_path_p.read_text())

        LOGGER.info(cls.config)

        model_options = cls._create_options(
            cls, cls.config["models"], SelectionStrategy(cls.config["model_selection"])
        )  # type: ignore
        template_options = cls._create_options(
            cls,
            cls.config["prompt_templates"],
            SelectionStrategy(cls.config["template_selection"]),
        )  # type: ignore

        LOGGER.info(
            f"🤖 Loaded {len(model_options)} models and {len(template_options)} templates"
        )
        all_options = [
            (model, template)
            for model in model_options
            for template in template_options
        ]

        if "retrieval" in cls.config:
            retrieval_options = cls._create_options(
                cls,
                cls.config["retrieval"],
                SelectionStrategy(cls.config["retrieval_selection"]),
            )  # type: ignore
            all_options = [
                (model, template, retrieval)
                for model, template in all_options
                for retrieval in retrieval_options
            ]

        LOGGER.info(f"🤖 Created {len(all_options)} scenarios from config")
        LOGGER.info(f"🤖 Scenarios: {all_options}")

        cls._load_template_map(cls)  # type: ignore

        # TODO: is this same behaviour as orig?
        # param_list = product(*all_options)
        # param_list = [reduce(lambda x, y: x | y, params) for params in param_list]

        if "retrieval" in cls.config:
            return cls(
                [
                    Scenario(
                        model=model["model"],
                        generation_engine=model["generation_engine"],
                        retrieval_window=retrieval["retrieval_window"],
                        top_k_retrieval_results=retrieval["top_k"],
                        prompt=Prompt(
                            prompt_template=template["prompt_template"],
                            prompt_content=cls.template_map[
                                template["prompt_template"]
                            ],
                        ),
                        src_config=cls.config,
                    )
                    for model, template, retrieval in all_options
                ]
            )
        return cls(
            [
                Scenario(
                    model=model["model"],
                    generation_engine=model["generation_engine"],
                    prompt=Prompt(
                        prompt_template=template["prompt_template"],
                        prompt_content=cls.template_map[template["prompt_template"]],
                    ),
                    src_config=cls.config,
                )
                for model, template in all_options
            ]
        )

    def _create_options(
        self, all_options: list, selection_type: SelectionStrategy
    ) -> list:
        """Create a list of options for a given selection type"""
        if selection_type == SelectionStrategy.RANDOM:
            return [random.choice(all_options)]

        return all_options

    def save_to_file(self, queries: list[Query], output_file: str) -> None:
        """Writes the list of queries in the output file"""
        _open = self._get_file_opener(output_file)

        with _open(output_file, "w") as f:
            for q in queries:
                f.write(q.model_dump_json() + "\n")

    def load_seed_queries(self) -> list[Query]:
        """Loads the seed queries from based on the config file"""
        seed_queries_path = (
            self.config["seed_queries_path"]
            if self.config["seed_queries_path"].startswith("s3")
            else config.root.parent / self.config["seed_queries_path"]
        )
        print(f"🤖 Loading seed queries from {seed_queries_path}")
        _open = self._get_file_opener(seed_queries_path)

        with _open(seed_queries_path) as f:
            seed_queries: list[Query] = [
                Query(**json.loads(line)) for line in f.readlines() if line
            ]

        print(f"🤖 Loaded {len(seed_queries)} seed queries")
        return seed_queries

    def _get_file_opener(self, file_path: str):
        if file_path.startswith("s3://"):
            labs_key = json.loads(get_secret("LABS_CREDS"))

            return s3fs.S3FileSystem(
                key=labs_key["key"], secret=labs_key["secret"]
            ).open
        return open

    def _load_template_map(self) -> dict[str, jinja2.Template]:
        """Loads a map of `{"template_path": jinja2.Template}` from the config file."""
        self.template_map: dict[str, jinja2.Template] = {}

        for template_path in self.config["prompt_templates"]:
            _p = template_path["prompt_template"]
            self.template_map[_p] = jinja_template_loader(
                config.root_templates_folder / f"{_p}.txt"
            )

        return self.template_map

    def __iter__(self):
        """Allows iteration over the scenarios list."""
        assert hasattr(
            self, "scenarios"
        ), "😱 Attribute 'scenarios' not found in ScenarioController."
        return iter(self.scenarios)
