import json

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from cpr_data_access.models import BaseDocument

from src.models.data_models import Query, QueryType
from src.logger import get_logger
from src.config import VERTEX_MODEL_ENDPOINTS


LOGGER = get_logger(__name__)


@dataclass
class PromptTemplateArguments:
    """Arguments for the prompt template."""

    seed_queries: Optional[list[Query]]
    document: str
    rule: Optional[str]


def render_document_text_for_llm(document_text: str, model: str) -> str:
    """
    Truncates the document text based on the model's token limit.

    Uses fast approximation from https://stackoverflow.com/questions/76216113/how-can-i-count-tokens-before-making-api-call so can remove tiktoken => all the torch dependencies from docker image for size optimisation.
    """
    _encoded_doc_length = len(document_text) * (1 / 2.718281828) + 2

    if model in VERTEX_MODEL_ENDPOINTS:
        truncation_amnt = 1500
        max_tokens = (
            VERTEX_MODEL_ENDPOINTS[model]["params"]["max_tokens"]
            if "max_tokens" in VERTEX_MODEL_ENDPOINTS[model]["params"]
            else VERTEX_MODEL_ENDPOINTS[model]["params"]["max_output_tokens"]
        )

        if _encoded_doc_length > max_tokens:
            truncate_to = max_tokens - truncation_amnt
            LOGGER.warning(
                f"Document too long [{_encoded_doc_length}], truncating to {truncate_to} tokens."
            )
            return document_text[: int(truncate_to)]
        return document_text

    if _encoded_doc_length > 30000 and "gemini" not in model:
        LOGGER.warning(
            f"Document too long [{_encoded_doc_length}], truncating to 30k tokens."
        )

        if "16k" in model:
            return document_text[: int(14500 * 2.718281828)]
        else:
            return document_text[: int(30000 * 2.718281828)]

    return document_text


def parse_response_into_queries(
    response_text: str, document_id: str, prompt_template: str, model: str
) -> list[Query]:
    """Parses the json in the generation into list of queries."""
    response_text = sanitise_response_text(response_text)
    generations = json.loads(response_text)["queries"]
    queries = []
    for q in generations:
        queries.append(
            Query(
                text=q,
                type=QueryType.SYNTHETIC,
                timestamp=datetime.now(),
                model=model,
                prompt_template=prompt_template,
                document_id=document_id,
            )
        )
    return queries


def sanitise_response_text(response_text: str) -> str:
    """Sanitises the response text to make sure it's a pure json string."""
    return response_text.strip("```json").strip("```")


def render_document(document: BaseDocument) -> str:
    return "\n".join([tb.to_string() for tb in document.text_blocks])
