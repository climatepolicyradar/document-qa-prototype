# from guardrails import Guard, OnFailAction
# from guardrails.hub import ToxicLanguage, DetectPII, WebSanitization

from src import config  # needed to load secrets from AWS credentials manager
from src.logger import get_logger

import requests
from typing import Dict
from pydantic import BaseModel, Field

assert config

LOGGER = get_logger(__name__)


class GuardrailValidationRequest(BaseModel):
    """Request to validate text against guardrails."""

    text: str = Field(..., description="Text to validate against guardrails")


class GuardrailValidationResponse(BaseModel):
    """Response from guardrails API"""

    overall_result: bool
    individual_results: Dict[str, bool]


class GuardrailController:
    """Controller for interacting with the Guardrails API."""

    def __init__(
        self, api_url: str = "https://guardrails-api.labs.climatepolicyradar.org"
    ):
        """
        Initializes the GuardrailController with the given API URL.

        Args:
            api_url (str): The URL of the Guardrails API.
        """
        self.api_url = api_url
        LOGGER.info(f"🛡️ Initialized GuardrailController with API URL: {self.api_url}")

    def validate(self, text: str) -> GuardrailValidationResponse:
        """
        Validates the input text against all guardrails.

        Args:
            text (str): The text to validate.

        Returns:
            GuardrailValidationResponse: The validation results.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the API response is invalid.
        """
        LOGGER.info(f"🔍 Validating text against guardrails: {text[:50]}...")

        try:
            response = requests.post(
                f"{self.api_url}/validate",
                json=GuardrailValidationRequest(text=text).model_dump(),
            )
            response.raise_for_status()
            data = response.json()

            validation_response = GuardrailValidationResponse(**data)
            LOGGER.info(
                f"✅ Guardrail validation complete. Overall result: {validation_response.overall_result}"
            )
            return validation_response

        except requests.RequestException as e:
            LOGGER.error(f"❌ Guardrail API request failed: {str(e)}")
            raise
        except ValueError as e:
            LOGGER.error(f"❌ Invalid response from Guardrail API: {str(e)}")
            raise

    def __str__(self) -> str:
        """Returns a string representation of the GuardrailController."""
        return f"GuardrailController(api_url={self.api_url})"


assert GuardrailController, "GuardrailController class must be defined"
