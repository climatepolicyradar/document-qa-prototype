import click

from typing import Union

from src.dataset_creation.argilla import (
    initialise_argilla,
    pull_annotations_and_push_to_hub,
    ArgillaConfig,
)
from src.dataset_creation.utils import get_config


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=str,
    help="Path to the configuration file",
    required=True,
)
@click.option(
    "--batch-name",
    "-b",
    type=str,
    help="Name of the batch(es) to pull annotations from",
    required=True,
    multiple=True,
)
@click.option(
    "--hub-repo-id",
    "-r",
    type=str,
    help="Hub repo id to push the annotations to",
    default="ClimatePolicyRadar/annotation-responses-unece",
)
def push_annotations_to_hub(
    config_path: str,
    batch_name: Union[str, list[str]],
    hub_repo_id: str,
):
    """Pulls the annotations from Argilla, merges them and publishes to the HF hub"""
    initialise_argilla()

    config = ArgillaConfig.model_validate(get_config(config_path))
    pull_annotations_and_push_to_hub(config, batch_name, hub_repo_id)


if __name__ == "__main__":
    push_annotations_to_hub()
