from src.models.data_models import EndToEndGeneration
from src.prompts.template_building import jinja_template_loader
from src.controllers.EvaluationController import evaluators
from src.evaluation.g_eval import GEval
from src.logger import get_logger

import src.config as config

LOGGER = get_logger(__name__)


@evaluators.register("coherence")
class Coherence(GEval):
    """
    G-Eval for coherence

    Based on the survey paper "[Leveraging Large Language Models for NLG Evaluation: Advances and Challenges](https://arxiv.org/pdf/2401.07103)"
    G-Eval is the most performant evaluation metric for coherence.

    The template, as with other G-eval evaluators, is a modified version of that from the original [G-eval paper](https://arxiv.org/pdf/2303.16634).
    """

    TYPE = "coherence"
    NAME = "g_eval"

    def __init__(self, *args):
        super().__init__(*args)
        self.template = jinja_template_loader(
            config.evaluation_templates_folder / "g_eval_coherence.txt"
        )

    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""
        return self.template.render(
            sources=generation.rag_response.retrieved_passages_as_string(),  # type: ignore
            question=generation.rag_request.query,
            answer=generation.rag_response.text,  # type: ignore
        )
