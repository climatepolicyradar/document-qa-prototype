from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from functools import lru_cache
from typing import Optional
import tiktoken

from src.logger import get_logger

LOGGER = get_logger(__name__)


def text_is_too_long_for_model(text: str, model_name: Optional[str] = None) -> bool:
    """Check if the text is too long for the model."""
    # This is in try/except loop as the default model used is gated, and needs to be approved before accessing.
    """
    TODO I'm writing this at 22:43 on saturday night so fairly sure there's a better way to do this
    try:
        context_length, tokenizer = get_context_length_and_tokenizer(model_name)
        return len(tokenizer.tokenize(text)) > context_length
    except Exception as e:
        LOGGER.error(f"Error checking text length: {e}")
        return False
    """

    enc_gpt_4 = tiktoken.encoding_for_model("gpt-4-32k")
    _encoded_doc = enc_gpt_4.encode(text)

    return len(_encoded_doc) > 30000


@lru_cache(maxsize=1024)
def get_context_length_and_tokenizer(
    model_name: Optional[str]
) -> tuple[int, AutoTokenizer]:
    """Get the context length and tokenizer for a given model."""
    if model_name is None:
        model_name = (
            "meta-llama/Meta-Llama-3-8B"  # TODO update this to the appropriate model
        )
        # Adding this for now, as it is using Tiktoken, which most of our LLM options will use
    config = PretrainedConfig.from_pretrained(model_name)
    context_length = config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return context_length, tokenizer
