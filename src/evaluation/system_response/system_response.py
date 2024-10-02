from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.controllers.EvaluationController import evaluators
from src.models.data_models import Score


@evaluators.register("system_response")
class SystemResponse(Evaluator):
    """Whether the system responded to the user query, based on substring matches."""

    TYPE = "system_response"
    NAME = "substring_match"

    @property
    def no_response_substrings(self) -> set[str]:
        """The substrings that indicate that the system did not respond."""

        return {
            "I cannot provide an answer",
        }

    def get_success(self, score: float) -> bool:
        """Returns whether the score is a success for this evaluator"""
        return score == 1

    def evaluate(self, generation: EndToEndGeneration) -> Score:
        """
        Evaluate whether the system responded.

        Returns a score of 1 if the system responded, 0.5 if the system said it wouldn't
        respond but seemed to then go on and provide context, and 0 otherwise.
        """
        self._validate_generation(generation)

        if generation.rag_response is None:
            return Score(
                score=0,
                type=self.TYPE,
                name=self.NAME,
                gen_uuid=generation.uuid,  # type: ignore
            )

        answer = generation.rag_response.text  # type: ignore

        negation_terms = {"but", "however"}

        if any(
            substring.lower() in answer.lower()
            for substring in self.no_response_substrings
        ):
            if any(term in answer.lower() for term in negation_terms):
                # If the system said it wouldn't respond but also used a negation term, it's
                # likely that it said "I cannot provide an answer, but here's {some
                # context}"

                # TODO: we could tokenise the answer for more robust checking of the
                # negation term here, but the risk of FPs seems low.
                return Score(
                    score=0.5,
                    success=self.get_success(0.5),
                    type=self.TYPE,
                    name=self.NAME,
                    gen_uuid=generation.uuid,  # type: ignore
                )
            else:
                return Score(
                    score=0,
                    success=self.get_success(0),
                    type=self.TYPE,
                    name=self.NAME,
                    gen_uuid=generation.uuid,  # type: ignore
                )

        return Score(
            score=1,  # type: ignore
            success=self.get_success(1),
            type=self.TYPE,
            name=self.NAME,
            gen_uuid=generation.uuid,  # type: ignore
        )
