import json
from typing import Optional
from src.evaluation.evaluator import Evaluator, Score
from src.models.data_models import EndToEndGeneration
from src.prompts.template_building import jinja_template_loader
from src.controllers.EvaluationController import evaluators
from src.logger import get_logger
from src.online.inference import get_llm


import src.config as config

LOGGER = get_logger(__name__)


@evaluators.register("patronus_lynx")
class PatronusLynx(Evaluator):
    """PatronusLynx model for Faithfulness"""

    TYPE = "faithfulness"
    NAME = "patronus_lynx"

    def __init__(self, *args):
        super().__init__(*args)
        self.model = get_llm("vertexai", "patronus-lynx")
        self.template = jinja_template_loader(
            config.evaluation_templates_folder / "patronus_lynx.txt"
        )

    def get_success(self, score: float) -> bool:
        """Returns whether the score is a success for this evaluator"""
        return score == 1

    def evaluate(
        self, generation: EndToEndGeneration, prompt: Optional[str] = None
    ) -> Optional[Score]:
        """Evaluates the generation"""

        if prompt is None:
            prompt = self.get_prompt(generation)

        response = self.model.invoke(prompt)
        result = {}
        try:
            result = json.loads(response.strip())
        except json.JSONDecodeError as e:
            LOGGER.error(f"Error decoding JSON: {result}\n\n {e}")
            return None

        return Score(
            score=1 if result["SCORE"] == "PASS" else 0,
            success=self.get_success(1 if result["SCORE"] == "PASS" else 0),
            type=self.TYPE,
            name=self.NAME,
            gen_uuid=generation.uuid or "",
            comments=result["REASONING"],
        )

    def get_prompt(self, generation: EndToEndGeneration) -> str:
        """Returns the prompt for the evaluator"""
        return self.template.render(
            context=generation.rag_response.retrieved_passages_as_string(),  # type: ignore
            question=generation.rag_request.query,
            answer=generation.get_answer(),  # type: ignore
        )
