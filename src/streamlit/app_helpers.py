import base64
import re


def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def _find_source_indices_in_rag_response(rag_response: str) -> list[int]:
    """
    Find the sources in a RAG response.

    These are numbers in square brackets, which could be comma-separated.
    E.g. [1] or [1, 2, 3].

    Returns a list of numbers of the sources in the response. Subtract 1 from each
    number to get the index in the list of sources.
    """

    pattern = r"\[(.*?)\]"
    numbers = re.findall(pattern, rag_response)
    numbers = " ".join(numbers)
    numbers = numbers.replace(",", " ")
    numbers = [int(n) for n in numbers.split() if n.isdigit()]

    unique_numbers = list(set(numbers))

    return unique_numbers


def _make_text_safe_for_streamlit_write(text: str) -> str:
    """
    Make text safe for Streamlit's write method.

    Dollar signs are used for LaTeX in Streamlit, so we need to escape them.
    """

    text = text.replace("$", "\$")  # type: ignore

    return text


def _format_window_as_source(window: list[dict]) -> str:
    """Formats the retrieved window into the standard source format for the app."""
    source = ""
    for doc in window:
        source += f"{doc['page_content']}\n"  #  type: ignore

    return source
