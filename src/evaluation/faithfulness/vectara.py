from typing import Optional
import torch

from transformers import AutoModelForSequenceClassification

from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.controllers.EvaluationController import evaluators
from src.evaluation.evaluator import Score


@evaluators.register("vectara")
class Vectara(Evaluator):
    """Vectara"""

    TYPE = "faithfulness"
    NAME = "vectara"
    MODEL_NAME = "vectara/hallucination_evaluation_model"
    MODEL_REVISION = "ade58fc7b0eeb92bac9bf2be0bbafdb1fd51d04a"  # vectara have pushed breaking changes in the past, fixing to commit hash

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,  # they have config loading code in their repo at the above revision that needs to be trusted to proceed
        )

    def evaluate(self, generation: EndToEndGeneration) -> Optional[Score]:
        """Evaluates the data"""
        self._validate_generation(generation)

        if generation.rag_response is None:
            return None

        context = generation.rag_response.retrieved_passages_as_string()  # type: ignore
        response = generation.rag_response.text  # type: ignore

        pairs = zip([context], [response])

        with torch.no_grad():
            scores = self.model.predict(pairs).detach().cpu().numpy()

        return Score(
            score=float(outputs[0]),
            type=self.TYPE,
            name=self.NAME,
            gen_uuid=generation.uuid,  # type: ignore
        )
