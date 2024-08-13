from enum import Enum
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, DetectPII, WebSanitization


class GuardrailType(Enum):
    """Available guardrail types"""

    TOXICITY = "toxicity"
    PII = "pii"
    WEB_SANITIZATION = "web_sanitization"


class GuardrailController:
    """Create multiple guardrails and use these to validate text."""

    guardrails: dict[GuardrailType, Guard]

    def __init__(
        self,
        guardrail_types: list[GuardrailType] = [
            GuardrailType.TOXICITY,
            GuardrailType.PII,
            GuardrailType.WEB_SANITIZATION,
        ],
    ):
        # At the moment we hardcode guardrails-ai to not raise exceptions, preferring to handle output objects instead
        self._on_fail_action = OnFailAction.NOOP
        print("Initialising guardrails...")
        self.guardrails: dict[GuardrailType, Guard] = {
            guardrail_type: self.create_guardrail(guardrail_type)
            for guardrail_type in guardrail_types
        }

    def _create_web_sanitization_guard(self) -> Guard:
        return Guard().use(WebSanitization, on_fail=self._on_fail_action)

    def _create_toxicity_guard(self, threshold: float = 0.5) -> Guard:
        return Guard().use(
            ToxicLanguage,
            threshold=threshold,
            validation_method="sentence",
            on_fail=self._on_fail_action,
        )

    def _create_pii_guard(self) -> Guard:
        """Create PII guard with default entity types"""

        entity_types = [
            "CREDIT_CARD",
            "CRYPTO",
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "PHONE_NUMBER",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "UK_NHS",
            "AU_ABN",
            "AU_ACN",
            "AU_TFN",
            "AU_MEDICARE",
            "IN_PAN",
            "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION",
            "IN_VOTER",
            "IN_PASSPORT",
        ]

        return Guard().use(DetectPII, entity_types, on_fail=self._on_fail_action)

    def create_guardrail(self, guardrail_type: GuardrailType, **kwargs) -> Guard:
        """Get a guardrail based on the type"""
        if guardrail_type == GuardrailType.TOXICITY:
            return self._create_toxicity_guard(**kwargs)
        elif guardrail_type == GuardrailType.PII:
            return self._create_pii_guard(**kwargs)
        elif guardrail_type == GuardrailType.WEB_SANITIZATION:
            return self._create_web_sanitization_guard(**kwargs)

    def _validate_text_individual_guardrail(self, text: str, guardrail: Guard) -> bool:
        if self._on_fail_action != OnFailAction.NOOP:
            raise NotImplementedError(
                "On fail action other than NOOP is not supported for individual guardrail validation"
            )

        result = guardrail.validate(text)

        return result.validation_passed

    def validate_text(self, text: str) -> tuple[bool, dict[GuardrailType, bool]]:
        """
        Validate text against all guardrails

        :param str text: text to validate
        :return tuple[bool, dict[GuardrailType, bool]]: overall validation result and individual guardrail results by name
        """
        individual_results = {
            guardrail_type: self._validate_text_individual_guardrail(text, guardrail)
            for guardrail_type, guardrail in self.guardrails.items()
        }

        overall_result = all(individual_results.values())

        return overall_result, individual_results
