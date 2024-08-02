import pytest

from src.evaluation.formatting import Formatting


@pytest.mark.parametrize(
    ("answer", "context", "output"),
    (
        ("This is a test answer without any quotations", "Test context", True),
        (
            'This is an answer with "1 quotation" and "another quotation"',
            "Test context",
            False,
        ),
        (
            'This is an answer with "1 quotation" and "another quotation"',
            'This is the context with "1 quotation" and "another quotation"',
            True,
        ),
    ),
)
def test_quotation_check(answer: str, context: str, output: bool):
    """Test that the quotation check returns True when there are no quotations"""
    formatting = Formatting()
    assert formatting.quotations_are_verbatim(answer, context) is output


@pytest.mark.parametrize(
    ("answer", "output"),
    (
        ("This is an answer in English", True),
        ("Ceci est un passage en française.", False),
    ),
)
def test_answer_is_english(answer: str, output: bool):
    """Test that the language check returns True when the answer is in English"""
    formatting = Formatting()
    assert formatting.answer_is_english(answer) is output


@pytest.mark.parametrize(
    ("answer", "citation_numbers", "output"),
    (
        (
            "This is the first sentence. [1] This is the second sentence [2]",
            {1, 2, 3},
            (True, None),
        ),
        (
            "This is the first sentence. This is the second sentence [1]",
            {1, 2},
            (False, "no_citation"),
        ),
        (
            "This is the first sentence that's an introduction:\n- first bulletpoint [1]\n- second bulletpoint [2]",
            {1, 2},
            (True, None),
        ),
        (
            "This is the first sentence. [5] This is the second. [2]",
            {1, 2},
            (False, "fictitious_citation"),
        ),
        (
            "This is a sentence with multiple citations. [1, 2] This is one with only one. [3]",
            {1, 2, 3},
            (True, None),
        ),
    ),
)
def test_sentences_followed_by_citations(
    answer: str, citation_numbers: set[int], output: bool
):
    """Test that the citation check returns True when each sentence is followed by a citation"""
    formatting = Formatting()
    assert (
        formatting.sentences_followed_by_citations(answer, citation_numbers) == output
    )
