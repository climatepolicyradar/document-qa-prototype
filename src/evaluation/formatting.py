import re

from ast import literal_eval
from typing import Optional
from langdetect import detect, LangDetectException
from nltk.tokenize import sent_tokenize

from src.evaluation.evaluator import Evaluator
from src.models.data_models import EndToEndGeneration
from src.controllers.EvaluationController import evaluators
from src.evaluation.evaluator import Score
from src.logger import get_logger

logger = get_logger(__name__)


@evaluators.register("formatting")
class Formatting(Evaluator):
    """Whether the system responded to the user query, based on substring matches."""

    TYPE = "formatting"
    NAME = "rule_based"

    def evaluate(self, generation: EndToEndGeneration) -> Score:
        """Evaluate whether the system followed the formatting guidelines"""
        self._validate_generation(generation)

        if generation.rag_response is None:
            return Score(
                score=0,
                type=self.TYPE,
                name=self.NAME,
                gen_uuid=generation.uuid,  # type: ignore
                comments=["rag_response_is_none"],
            )

        answer = generation.rag_response.text  # type: ignore
        citation_numbers = generation.rag_response.citation_numbers  # type: ignore

        well_formatted = True
        comments = []

        if not self.quotations_are_verbatim(
            answer,
            generation.rag_response.retrieved_passages_as_string(),  # type: ignore
        ):
            well_formatted = False
            comments.append("quotations_not_verbatim")

        # short answers trigger an error or confuse language detection
        if len(answer.strip()) > 50 and (not self.answer_is_english(answer)):
            well_formatted = False
            comments.append("answer_not_english")

        citations_ok, citation_comment = self.sentences_followed_by_citations(
            answer, citation_numbers
        )

        if not citations_ok:
            well_formatted = False
            comments.append(citation_comment)

        return Score(
            score=int(well_formatted),  # type: ignore
            type=self.TYPE,
            name=self.NAME,
            gen_uuid=generation.uuid,  # type: ignore
            comments=comments if comments else None,
        )

    def quotations_are_verbatim(self, answer: str, context: str) -> bool:
        """Check if the answer contains verbatim quotations from the context"""
        quotations = re.findall(r'"([^"]*)"', answer)

        for quotation in quotations:
            if quotation.lower() not in context.lower():
                return False

        return True

    def answer_is_english(self, answer: str) -> bool:
        """Check if the answer is in English"""
        try:
            if detect(answer) == "en":
                return True
        except LangDetectException:
            logger.warning(f"Could not detect language for answer: {answer}")
        return False

    def sentences_followed_by_citations(
        self, answer: str, citation_numbers: set[int]
    ) -> tuple[bool, Optional[str]]:
        """
        Checks if each sentence / bulletpoint is followed by a citation

        NOTE: the current implementation will fail on no-response cases. This is accepted for now,
        because we have a no-response evaluator that will catch these cases, hence this should be used in
        conjunction with that.
        """
        sentences = sent_tokenize(answer)

        for idx, sentence in enumerate(sentences):
            # Sometimes NLTK splits the last citation into a separate sentence
            if len(sentence) < 10:
                continue

            start_char = answer.find(sentence) + len(sentence)
            following_string = answer[
                start_char - 20 : start_char + 20
            ]  # sometimes attached to the end of the sentence, sometimes the front

            brackets = re.findall(r"(\[[\d,\s]*\])", following_string)

            if not brackets:
                if idx == 0 and ":" in following_string and len(sentences) > 1:
                    # intro sentences followed by bulletpoints don't need citations
                    continue
                else:
                    return False, "no_citation"

            for bracket in brackets:
                cited_numbers = set(literal_eval(bracket))

                if cited_numbers.intersection(citation_numbers):
                    continue
                else:
                    return False, "fictitious_citation"

        return True, None
