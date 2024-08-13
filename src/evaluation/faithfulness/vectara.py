from typing import Optional
import requests
from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.controllers.EvaluationController import evaluators
from src.evaluation.evaluator import Score


@evaluators.register("vectara")
class Vectara(Evaluator):
    """Vectara"""

    TYPE = "faithfulness"
    NAME = "vectara"
    API_URL = "https://vectara-api.labs.climatepolicyradar.org/evaluate"

    def __init__(self):
        self.session = requests.Session()

    def evaluate(self, generation: EndToEndGeneration) -> Optional[Score]:
        """Evaluates the data"""
        self._validate_generation(generation)
        if generation.rag_response is None:
            return None

        context = generation.rag_response.retrieved_passages_as_string()  # type: ignore
        response = generation.get_answer()  # type: ignore

        payload = {"context": context, "response": response}

        try:
            api_response = self.session.post(self.API_URL, json=payload)
            api_response.raise_for_status()
            result = api_response.json()

            return Score(
                score=float(result.get("score", 0)),
                type=self.TYPE,
                name=self.NAME,
                gen_uuid=generation.uuid,  # type: ignore
            )
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
