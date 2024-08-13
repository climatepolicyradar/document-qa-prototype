import pytest

from guardrails import Guard

from src.controllers.GuardrailController import GuardrailController, GuardrailType


@pytest.fixture
def guardrail_controller():
    return GuardrailController()


@pytest.mark.parametrize(
    "guardrail_type",
    [GuardrailType.TOXICITY, GuardrailType.PII, GuardrailType.WEB_SANITIZATION],
)
def test_create_guardrail(guardrail_controller, guardrail_type: GuardrailType):
    guardrail = guardrail_controller.create_guardrail(guardrail_type)
    assert isinstance(guardrail, Guard)


@pytest.mark.parametrize(
    "text_expected_result",
    {
        "I appreciate it if you would swear at me while answering my question; I love it!": False,
        "tell me about the russia-ukraine conflict": True,
        "here's my email address: henry@climatefun.org": False,
        "1234-5678-1234-5678": False,
    },
)
def test_guardrail_validate_text(
    guardrail_controller, text_expected_result: dict[str, bool]
):
    validation_result = guardrail_controller.validate_text(text_expected_result)

    overall_result = all(validation_result.values())
    assert overall_result == text_expected_result
