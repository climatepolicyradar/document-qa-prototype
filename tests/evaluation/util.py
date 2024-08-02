import pytest

import pandas as pd

from src.models.data_models import EndToEndGeneration


@pytest.fixture
def e2e_data():
    generations_df = pd.read_json(
        "tests/evaluation/end_to_end_generations.jsonl", lines=True
    )

    end_to_end_generations = (
        EndToEndGeneration.model_validate(generation)
        for generation in generations_df["generation"].tolist()
    )

    return end_to_end_generations
