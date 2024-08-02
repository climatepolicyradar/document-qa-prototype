import jinja2
import tiktoken
import json

from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from langchain_core.messages.base import BaseMessage
from cpr_data_access.models import BaseDocument

from src.models.data_models import Query, QueryType
from src.online.inference import get_llm
from src.logger import get_logger


LOGGER = get_logger(__name__)
enc_gpt_4 = tiktoken.encoding_for_model("gpt-4-32k")


@dataclass
class PromptTemplateArguments:
    """Arguments for the prompt template."""

    seed_queries: Optional[list[Query]]
    document: str
    rule: Optional[str]

def render_document_text_for_llm(document_text: str, model: str) -> str:
    """Truncates the document text based on the model's token limit."""
    _encoded_doc = enc_gpt_4.encode(document_text)

    if len(_encoded_doc) > 30000 and "gemini" not in model:
        LOGGER.warning(
            f"Document too long [{len(_encoded_doc)}], truncating to 30k tokens."
        )

        if "16k" in model:
            return enc_gpt_4.decode(_encoded_doc[:14500])
        else:
            return enc_gpt_4.decode(_encoded_doc[:30000])

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
