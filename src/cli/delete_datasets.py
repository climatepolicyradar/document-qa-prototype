import typer
import argilla as rg

from src.logger import get_logger
from src.cli.load_generations_into_argilla import ArgillaConfig
from src.dataset_creation.utils import get_config

LOGGER = get_logger(__name__)


def delete_datasets_with_confirmation(
    dataset_names: list[str], config_path: str
) -> None:
    ARGILLA_CONFIG = ArgillaConfig.model_validate(get_config(config_path))

    for username in ARGILLA_CONFIG.users:
        for dataset_name in dataset_names:
            try:
                dataset = rg.FeedbackDataset.from_argilla(
                    name=dataset_name,
                    workspace=username,
                )

                delete_dataset = typer.confirm(
                    f"Delete dataset with {len(dataset.records)} records {dataset_name} from {username}?"
                )

                if delete_dataset:
                    dataset.delete()
                    LOGGER.info(f"Dataset {dataset_name} deleted.")

                else:
                    LOGGER.info(f"Dataset {dataset_name} not deleted.")

            except ValueError:
                LOGGER.info(f"Dataset {dataset_name} not found in {username}.")
                dataset = None


if __name__ == "__main__":
    typer.run(delete_datasets_with_confirmation)
