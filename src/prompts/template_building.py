from typing import Union
import jinja2

from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.config import (
    response_templates_folder,
    policy_templates_folder,
    root_templates_folder,
)

_cpr_generation_policy = (
    policy_templates_folder / "cpr_generation_policy_shortened.txt"
).read_text()
system_prompt: str = Path(response_templates_folder / "system_prompt.txt").read_text()


def jinja_template_loader(path: Union[str, Path]) -> jinja2.Template:
    """Loads jinja template from file."""
    with open(path, "r") as f:
        template = jinja2.Template(f.read())
    return template


def get_citation_template(template_name: str) -> PromptTemplate:
    template = (root_templates_folder / f"{template_name}.txt").read_text()

    if "{rag_policy}" in template:
        template = template.replace("{rag_policy}", _cpr_generation_policy)

    return PromptTemplate.from_template(template)


def make_qa_prompt(
    user_prompt_template: PromptTemplate, system_prompt: str
) -> ChatPromptTemplate:
    """
    Make a question answering prompt.

    TODO: this could use chat history too, if we extend to multi-turn.

    :param PromptTemplate user_prompt_template: prompt template used for user query
    :param str system_prompt: system prompt
    :return ChatPromptTemplate: template ready for langchain pipeline
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt_template.template),
        ]
    )
