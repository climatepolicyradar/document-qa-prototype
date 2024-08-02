# Prompt templates

This folder contains the templates that we use for LLM prompting. They are broken down into folders by use case.

## Query

These are prompt templates used for generating queries. These are jinja2 templates.

## Response

Prompts used for response generation, including adversarial, system-prompt and the RAG pipeline prompt.

## Policy

The cpr-generation-policy is stored here. **This is not a prompt that will be used in the RAG system**, instead it's used for redteaming and for labelling documentation. Hence, it's a bit more verbose and talks about 'the system' rather than 'you' when describing what the system should do.
