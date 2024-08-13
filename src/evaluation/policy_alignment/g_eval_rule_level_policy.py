from src.models.data_models import EndToEndGeneration
from src.prompts.template_building import jinja_template_loader
from src.evaluation.util import evaluators
from src.evaluation.evaluator import Evaluator
from src.logger import get_logger
from src.evaluation.policy_alignment.g_eval_policy import GEvalPolicy
from src.evaluation.evaluator import Score

import src.config as config

LOGGER = get_logger(__name__)


@evaluators.register("g_eval_rule_level_policy")
class GEvalRuleLevelPolicy(Evaluator):
    """
    G-Eval for policy aligment using a rule-level evaluation

    This evaluator is different from the rest in that it outputs n scores, where n is
    the number of rules in the generation policy.

    The names assigned to the scores are "g_eval-rule-0", "g_eval-rule-1", etc.
    """

    TYPE = "cpr-generation-policy"
    NAME = ""

    def __init__(self, *args):
        super().__init__(*args)
        self.policy = (
            config.policy_templates_folder / "cpr_generation_policy_general.txt"
        ).read_text()
        self.template = jinja_template_loader(
            config.evaluation_templates_folder
            / "cpr-generation-policy/g_eval_rule_level_alignment.txt"
        )
        self.rules = self.policy.split("\n- ")[1:]
        self.evaluators = [
            GEvalPolicy(name=f"g_eval-rule-{idx}") for idx, _ in enumerate(self.rules)
        ]

    def evaluate(self, generation: EndToEndGeneration) -> list[Score]:
        """Evaluates the generation"""
        self._validate_generation(generation)

        if generation.rag_response is None:
            return []

        scores = []

        for evaluator, rule in zip(self.evaluators, self.rules):
            score = evaluator.evaluate(
                generation, prompt=self.get_prompt(generation, rule)
            )
            if score is not None:
                scores.append(score)

        return scores

    def get_prompt(self, generation: EndToEndGeneration, rule: str) -> str:
        """Returns the prompt for the evaluator"""
        return self.template.render(
            rule=rule,
            sources=generation.rag_response.retrieved_passages_as_string(),  # type: ignore
            question=generation.rag_request.query,
            answer=generation.rag_response.text,  # type: ignore
        )
