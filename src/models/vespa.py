
from typing import Optional
from src.vespa.connection import connect_to_vespa_cloud
from src.vespa.query import get_rank_profiles, make_request
from pydantic import BaseModel
from vespa.application import Vespa
from src.logger import get_logger
import argilla as rg


LOGGER = get_logger(__name__)

def get_vespa_app() -> Vespa:
    LOGGER.info("Connecting to Vespa Cloud...")
    vespa_app = connect_to_vespa_cloud()
    LOGGER.info("Success")
    return vespa_app
    

class VespaTextBlock(BaseModel):
    """Attributes of text blocks in Vespa"""

    text_block_id: str
    text_block: str
    text_block_window: str


class VespaRankingResult(BaseModel):
    """
    One result returned by Vespa.

    Rank is 0-indexed.
    """

    text_block: VespaTextBlock
    score: float
    rank: int
    ranking_features: dict = {}


class VespaResponse(BaseModel):
    """A complete response from Vespa, with some useful request parameters."""

    query: str
    document_id: str
    rank_profile: str
    results: list[VespaRankingResult]

    def to_argilla_records(self, num_sample: int) -> list[rg.FeedbackRecord]:
        """
        Convert the Vespa response to a list of Argilla FeedbackRecord objects.

        Does this by sampling `num_sample` results from the Vespa response. A random
        seed should be set if you want this to be reproducible.

        :raises ValueError: if `num_sample` is greater than the number of results in the Vespa response.
        """
        if len(self.results) < num_sample:
            raise ValueError(
                f"Number of samples to take ({num_sample}) must be less than the number of results ({len(self.results)})"
            )

        results_sample = random.sample(self.results, num_sample)

        feedback_records = []

        for result in results_sample:
            feedback_records.append(
                rg.FeedbackRecord(
                    fields={
                        "query": self.query,
                        "text_block": result.text_block.text_block,
                        "text_block_window": result.text_block.text_block_window,
                    },
                    metadata={
                        "document_id": self.document_id,
                        "rank_profile": self.rank_profile,
                        "score": result.score,
                        "rank": result.rank,
                    },
                )
            )

        return feedback_records


def vespa_response_is_valid(response: dict) -> bool:
    """
    Check whether a Vespa response contains the expected fields.

    If not, it's likely because the document ID requested is not in the Vespa index, so
    the response parsing should be skipped.
    """

    return "root" in response and "children" in response["root"]


def process_vespa_response(response: dict) -> Optional[list[VespaRankingResult]]:
    """
    Process a Vespa response into a list of VespaRankingResult objects.

    Returns None if the response is invalid, which likely means the document ID in
    the request is not in Vespa.
    """

    if not vespa_response_is_valid(response):
        return None

    processed_results = []

    for idx, result in enumerate(response["root"]["children"]):
        text_block = VespaTextBlock(
            text_block_id=result["fields"]["text_block_id"],
            text_block=result["fields"]["text_block"],
            text_block_window=result["fields"]["text_block_window"],
        )

        score = result["relevance"]

        ranking_features = result["fields"]["matchfeatures"]

        processed_results.append(
            VespaRankingResult(
                text_block=text_block,
                score=score,
                rank=idx,
                ranking_features=ranking_features,
            )
        )

    return processed_results
