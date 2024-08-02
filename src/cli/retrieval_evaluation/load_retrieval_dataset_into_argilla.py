from pathlib import Path
import random
from typing import Optional

import typer
import argilla as rg

from src.dataset_creation.argilla import (
    ArgillaConfig,
    initialise_argilla,
    create_or_get_workspace_with_user_access,
    get_relevance_questions,
)
from src.dataset_creation.utils import get_config
from src.logger import get_logger
from src.cli.retrieval_evaluation.create_retrieval_dataset import VespaResponse

LOGGER = get_logger(__name__)


def load_retrieval_dataset(dataset_path: str) -> list[VespaResponse]:
    """Load a retrieval dataset from a JSONL file."""
    LOGGER.info(f"Loading retrieval dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        retrieval_dataset = f.readlines()
    return [
        VespaResponse.model_validate_json(response) for response in retrieval_dataset
    ]


def get_relevance_feedback_dataset(
    guidelines: Optional[str],
):
    """Get an empty relevance feedback dataset for Argilla."""

    relevance_questions = get_relevance_questions()

    return rg.FeedbackDataset(
        guidelines=guidelines,
        fields=[
            rg.TextField(name="query", title="Question / query to the system"),
            rg.TextField(
                name="text_block", title="Retrieved text block", use_markdown=True
            ),
            rg.TextField(
                name="text_block_window",
                title="Window around text block",
                use_markdown=True,
            ),
        ],
        questions=relevance_questions,  # type: ignore
    )


def load_retrieval_data_into_argilla(
    retrieval_dataset_jsonl_path: str,
    argilla_dataset_name: str,
    num_records_to_annotate: int,
    argilla_records_per_query: int = 3,  # TODO: maybe put this in config
    config_path: Path = Path(
        "src/dataset_creation/configs/argilla_config_relevance.yaml"
    ),
    random_seed: int = 42,
):
    """
    Load retrieval dataset for labelling into Argilla.

    Note this CLI behaves differently to the Argilla CLI to load generations. Instead
    of assigning records with an overlap, this limits the number of records to load
    and then loads them into a workspace shared by all users in the config.
    """
    random.seed(random_seed)

    LOGGER.info(f"Loading config from {config_path}")
    ARGILLA_CONFIG = ArgillaConfig.model_validate(get_config(config_path))

    if ARGILLA_CONFIG.annotations_per_record != len(ARGILLA_CONFIG.users):
        raise ValueError(
            "Argilla config `annotations_per_record` must be equal to the number of users. See CLI help for more information."
        )

    dataset_to_load = load_retrieval_dataset(retrieval_dataset_jsonl_path)
    LOGGER.info(
        f"Loaded {len(dataset_to_load)} records from {retrieval_dataset_jsonl_path}"
    )

    argilla_records = []
    for record in dataset_to_load:
        argilla_records.extend(
            record.to_argilla_records(num_sample=argilla_records_per_query)
        )

    if num_records_to_annotate > len(argilla_records):
        LOGGER.warning(
            f"Number of records to annotate ({num_records_to_annotate}) cannot be greater than the number of records in the dataset ({len(argilla_records)}). Increase the number of argilla records per query if you'd like more to annotate. Continuing with {len(argilla_records)} records."
        )

    records_to_annotate = random.sample(argilla_records, num_records_to_annotate)
    LOGGER.info(f"Sampled {len(records_to_annotate)} records.")

    LOGGER.info("Initialising Argilla connection")
    initialise_argilla()

    # This assertion shouldn't be triggered as the ArgillaConfig model validates this
    assert (
        ARGILLA_CONFIG.workspace_name is not None
    ), "Argilla config `workspace_name` must be provided."

    argilla_workspace = create_or_get_workspace_with_user_access(
        ARGILLA_CONFIG.workspace_name, ARGILLA_CONFIG.users
    )
    LOGGER.info(
        f"Created workspace {ARGILLA_CONFIG.workspace_name} with users {ARGILLA_CONFIG.users}"
    )

    typer.confirm(
        f"{len(records_to_annotate)} records ready to load into Argilla workspace {ARGILLA_CONFIG.workspace_name}. Continue?",
        abort=True,
    )

    argilla_dataset = get_relevance_feedback_dataset(guidelines=None)
    argilla_dataset.add_records(records_to_annotate)

    argilla_dataset.push_to_argilla(
        name=argilla_dataset_name, workspace=argilla_workspace
    )


if __name__ == "__main__":
    typer.run(load_retrieval_data_into_argilla)
