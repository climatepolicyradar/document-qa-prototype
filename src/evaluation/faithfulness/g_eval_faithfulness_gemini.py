from src.evaluation.g_eval import GEval
from src.models.data_models import EndToEndGeneration
from src.prompts.template_building import jinja_template_loader
from src.controllers.EvaluationController import evaluators
from src.logger import get_logger
from src.online.inference import get_llm


import src.config as config


LOGGER = get_logger(__name__)


@evaluators.register("g_eval_faithfulness_gemini")
class GEvalFaithfulness(GEval):
    """G-Eval for Faithfulness"""

    TYPE = "faithfulness_gemini"
    NAME = "g_eval"

    def __init__(self, *args):
        super().__init__(*args)

        self.model = get_llm("gemini", "gemini-1.5-pro")
        self.template = jinja_template_loader(
            config.evaluation_templates_folder / "g_eval_faithfulness.txt"
        )

    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""

        result = self.template.render(
            sources=generation.rag_response.retrieved_passages_as_string(),  # type: ignore
            question=generation.rag_request.query,
            answer=generation.get_answer(),  # type: ignore
        )
        return result

    def get_success(self, score: float) -> bool:
        """Returns whether the score is a success for this evaluator"""
        return score >= 0.8
