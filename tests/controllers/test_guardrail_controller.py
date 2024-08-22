import pytest

from guardrails import Guard

from src.controllers.GuardrailController import GuardrailController, GuardrailType


@pytest.fixture
def guardrail_controller():
    return GuardrailController()


@pytest.mark.parametrize(
    "guardrail_type",
    [
        GuardrailType.TOXICITY,
        # GuardrailType.PII, # This fails in CI as the original package is broken
        GuardrailType.WEB_SANITIZATION,
    ],
)
def test_create_guardrail(guardrail_controller, guardrail_type: GuardrailType):
    guardrail = guardrail_controller.create_guardrail(guardrail_type)
    assert isinstance(guardrail, Guard)


@pytest.mark.parametrize(
    "input_text_and_expected_result",
    [
        ("you are a stupid silly idiot!", False),
        ("tell me about the russia-ukraine conflict", True),
        ("here's my email address: henry@climatefun.org", False),
        ("sure my phone number is 07928 943 934", False),
    ],
)
def test_guardrail_validate_text(
    guardrail_controller, input_text_and_expected_result: list[tuple[str, str]]
):
    input_text, expected_result = input_text_and_expected_result
    overall_result, validation_results = guardrail_controller.validate_text(input_text)

    assert (
        overall_result == expected_result
    ), f"Test failed for {input_text}. Breakdown: {validation_results}"
    assert overall_result == all(validation_results.values())
