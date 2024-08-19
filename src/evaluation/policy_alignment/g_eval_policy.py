from src.models.data_models import EndToEndGeneration
from typing import Optional
from src.prompts.template_building import jinja_template_loader
from src.controllers.EvaluationController import evaluators
from src.evaluation.g_eval import GEval
from src.logger import get_logger

import src.config as config

LOGGER = get_logger(__name__)


@evaluators.register("g_eval_policy")
class GEvalPolicy(GEval):
    """G-Eval for policy aligment"""

    TYPE = "cpr-generation-policy"
    NAME = "g_eval"

    def __init__(self, *args, name: Optional[str] = None):
        super().__init__(*args)
        self.policy = (
            config.policy_templates_folder / "cpr_generation_policy_general.txt"
        ).read_text()
        self.template = jinja_template_loader(
            config.evaluation_templates_folder
            / "cpr-generation-policy/g_eval_violation_v5.txt"
        )
        if name is not None:
            self.NAME = name

    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""
        return self.template.render(
            policy=self.policy,
            sources=generation.rag_response.retrieved_passages_as_string(),  # type: ignore
            question=generation.rag_request.query,
            answer=generation.get_answer(),  # type: ignore
        )
