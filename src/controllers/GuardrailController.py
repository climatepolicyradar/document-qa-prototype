from enum import Enum
# from guardrails import Guard, OnFailAction
# from guardrails.hub import ToxicLanguage, DetectPII, WebSanitization

from src import config  # needed to load secrets from AWS credentials manager
from src.logger import get_logger

assert config

LOGGER = get_logger(__name__)


class GuardrailType(Enum):
    """Available guardrail types"""

    TOXICITY = "toxicity"
    PII = "pii"
    WEB_SANITIZATION = "web_sanitization"
