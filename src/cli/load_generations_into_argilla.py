from collections import defaultdict
from typing import Optional
from pathlib import Path
import random

import typer
import argilla as rg
from argilla.client.feedback.utils import assign_records, assign_workspaces
import pandas as pd

from src.dataset_creation.argilla import ArgillaConfig
from src.dataset_creation.utils import get_config, get_hash
from src.models.data_models import EndToEndGeneration, RAGResponse
from src.dataset_creation.argilla import (
    get_generation_policy_refinement_questions,
    get_evaluation_axes_questions,
    initialise_argilla,
)
from src.logger import get_logger

LOGGER = get_logger(__name__)


def get_empty_feedback_dataset(
    guidelines: Optional[str],
    task_type: str,
):
    if task_type == "generation_policy_refinement":
        questions = get_generation_policy_refinement_questions()
    elif task_type == "evaluation_axes":
        questions = get_evaluation_axes_questions()
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return rg.FeedbackDataset(
        guidelines=guidelines,
        fields=[
            rg.TextField(name="question", title="Question / query to the system"),
            rg.TextField(name="output", title="Generated output", use_markdown=True),
            rg.TextField(
                name="sources", title="Sources used for the response", use_markdown=True
            ),
        ],
        questions=questions,  # type: ignore
    )


def filter_number_of_no_response_generations(
    generations: list[EndToEndGeneration], max_proportion_answer_refused: float
) -> list[EndToEndGeneration]:
    """
    Filter a list of generations to a reasonable number of 'no response' answers.

    Max_proportion_answer_refused is the maximum proportion of refused answers that is
    acceptable out of the total number of generations.
    """

    generations_answer_refused = [
        generation
        for generation in generations
        if generation.rag_response.refused_answer()  # type: ignore
    ]

    generations_answer_accepted = [
        generation
        for generation in generations
        if not generation.rag_response.refused_answer()  # type: ignore
    ]

    total_num_generations = len(generations)

    # Check if the proportion of refused answers is within the acceptable range
    proportion_answer_refused = len(generations_answer_refused) / total_num_generations
    if proportion_answer_refused > max_proportion_answer_refused:
        # sample fewer refused answers
        n_samples = int(max_proportion_answer_refused * total_num_generations)
        LOGGER.info(
            f"Proportion of refused answers ({proportion_answer_refused:.2f}) is greater than the maximum allowed ({max_proportion_answer_refused:.2f})."
            f"Sampling {n_samples}/{len(generations_answer_refused)} no response cases. This will bring the total number of Argilla records to {len(generations_answer_accepted) + n_samples}."
        )
        generations_answer_refused = random.sample(
            generations_answer_refused, n_samples
        )

    return generations_answer_accepted + generations_answer_refused


def rag_response_text_is_valid_for_argilla(rag_response_text: str) -> bool:
    """
    Use heuristics to check for responses we don't want to show annotators.

    Returns True if we should show it to an annotator; False if not.
    """

    _phrases_to_exclude = {
        "i am a",
        "i am not a",
    }

    if any(phrase in rag_response_text.lower() for phrase in _phrases_to_exclude):
        return False

    return True


def rag_response_source_text_length_is_valid_for_argilla(
    rag_response: RAGResponse, lower_char_threshold=0, upper_char_threshold=2800
) -> bool:
    """Whether the source text is within a valid length range for Argilla."""

    windows_text_markdown_stripped = rag_response.retrieved_windows_as_string().replace(
        "**", ""
    )

    return (
        lower_char_threshold
        < len(windows_text_markdown_stripped)
        < upper_char_threshold
    )


def filter_out_generations_present_in_argilla(
    generations: list[EndToEndGeneration], argilla_datasets: list[str]
) -> list[EndToEndGeneration]:
    """
    Filter out generations that are already present in Argilla.

    Filtering is done baesd on the hash of the question and response text.
    """

    existing_datasets = []
    ds_map = defaultdict(list)

    for dataset_name in argilla_datasets:
        for user in rg.User.list():
            try:
                _dataset = rg.FeedbackDataset.from_argilla(
                    name=dataset_name, workspace=user.username
                )
                existing_datasets.append(_dataset)
                ds_map[user.username].append(_dataset.name)
            except ValueError:
                # LOGGER.warning(
                #     f"Dataset '{dataset_name}' not found for user '{user.username}'"
                # )
                # TODO: should we warn that a dataset doesn't exist for a user,
                # rather than failing silently?
                continue

    LOGGER.info(f"Found {len(existing_datasets)} existing Argilla datasets\n{ds_map}")

    query_response_hashes_in_argilla = set()
    for dataset in existing_datasets:
        query_response_hashes_in_argilla.update(
            (
                get_hash(f'{record.fields["question"]}_{record.fields["output"]}')
                for record in dataset.records
            )
        )

    LOGGER.info(f"Found {len(query_response_hashes_in_argilla)} records in Argilla")

    generations_not_already_in_argilla = []

    for generation in generations:
        if generation.rag_response is None:
            continue

        gen_hash = get_hash(
            f"{generation.rag_request.query}_{generation.rag_response.text}"
        )

        if gen_hash not in query_response_hashes_in_argilla:
            generations_not_already_in_argilla.append(generation)

    _n_filtered_out = len(generations) - len(generations_not_already_in_argilla)

    LOGGER.info(
        f"Filtered out {_n_filtered_out} generations already in Argilla. {len(generations_not_already_in_argilla)} generations remaining."
    )

    return generations_not_already_in_argilla


def load_generations_into_argilla(
    generations_jsonl_path: str,
    task_type: str,
    argilla_dataset_name: str,
    max_proportion_answer_refused: float = 0.2,
    annotation_guidelines_path: Optional[Path] = None,
    config_path: Path = Path("src/dataset_creation/configs/argilla_config.yaml"),
    num_records_per_user: Optional[int] = None,
    random_seed: int = 42,
):
    """Load generations from a file created using the create_generations CLI into Argilla."""

    random.seed(random_seed)

    LOGGER.info(f"Loading config from {config_path}")
    ARGILLA_CONFIG = ArgillaConfig.model_validate(get_config(config_path))

    if ARGILLA_CONFIG.annotations_per_record == len(ARGILLA_CONFIG.users):
        raise NotImplementedError(
            "The ability to assign every record to every user is not yet implemented. If you're seeing this method, consider PRing the small change to make this possible!"
        )

    LOGGER.info("Creating Argilla records from generations data")
    if task_type not in ["generation_policy_refinement", "evaluation_axes"]:
        raise ValueError(f"Invalid task type: {task_type}")

    annotation_guidelines = (
        annotation_guidelines_path.read_text() if annotation_guidelines_path else None
    )
    generations_df = pd.read_json(generations_jsonl_path, lines=True)

    end_to_end_generations = (
        EndToEndGeneration.model_validate(generation)
        for generation in generations_df["generation"].tolist()
    )

    valid_generations = [
        generation
        for generation in end_to_end_generations
        if not generation.error
        and generation.rag_response is not None
        and rag_response_text_is_valid_for_argilla(generation.rag_response.text)
    ]
    LOGGER.info(
        f"Loaded {len(valid_generations)} generations with no error out of {len(generations_df)} total"
    )

    valid_generations = [
        generation
        for generation in valid_generations
        if generation.rag_response
        and rag_response_source_text_length_is_valid_for_argilla(
            generation.rag_response
        )
    ]

    LOGGER.info(
        f"Filtered out generations with source text length outside of the valid range. {len(valid_generations)} generations remaining."
    )

    initialise_argilla()

    if ARGILLA_CONFIG.existing_argilla_datasets is not None:
        LOGGER.info(
            "Checking for existing datasets in Argilla to filter out generations already present."
        )
        valid_generations = filter_out_generations_present_in_argilla(
            valid_generations, ARGILLA_CONFIG.existing_argilla_datasets
        )

    valid_generations = filter_number_of_no_response_generations(
        valid_generations, max_proportion_answer_refused
    )

    feedback_records: list[rg.FeedbackRecord] = [
        generation.to_argilla_feedback_record() for generation in valid_generations
    ]

    if num_records_per_user:
        num_records_sample = int(
            num_records_per_user
            * len(ARGILLA_CONFIG.users)
            / ARGILLA_CONFIG.annotations_per_record
        )
        if num_records_sample <= len(feedback_records):
            LOGGER.info(
                f"Sampling {num_records_per_user} records per user ({num_records_sample} total)"
            )
            feedback_records = random.sample(feedback_records, num_records_sample)

        else:
            LOGGER.warning(
                f"Number of records per user ({num_records_sample}) is greater than the number of available records ({len(feedback_records)}). Using all available records."
            )

    LOGGER.info("Assigning records to users")
    record_user_assignments = assign_records(
        users=ARGILLA_CONFIG.users,
        records=feedback_records,
        overlap=ARGILLA_CONFIG.annotations_per_record,
        shuffle=True,
    )

    # Assigns records to each user's personal workspace
    # Note the output is not important here, but this function does a bunch of user and
    # workspace validation checks behind the scenes
    record_workspace_assignments = assign_workspaces(
        assignments=record_user_assignments, workspace_type="individual"
    )
    # This step is needed to ensure that workspaces are assigned for 'owner' users.
    record_workspace_assignments = {
        u: v or [u] for u, v in record_workspace_assignments.items()
    }

    n_records_per_user = {
        user: len(records) for user, records in record_user_assignments.items()
    }

    LOGGER.info(
        f"Assigned {len(feedback_records)} records to {len(ARGILLA_CONFIG.users)} users. "
        f"Annotations per user: {n_records_per_user}"
    )

    typer.confirm("Continue?", default=None, abort=True)

    for username, records in record_user_assignments.items():
        feedback_dataset = get_empty_feedback_dataset(annotation_guidelines, task_type)
        feedback_dataset.add_records(records)

        for _workspace in record_workspace_assignments[username]:
            feedback_dataset.push_to_argilla(
                workspace=_workspace, name=argilla_dataset_name
            )


if __name__ == "__main__":
    typer.run(load_generations_into_argilla)
