"""
DEPRECATED

Use `src/flows/generate_answers_flow.py` instead.
"""

import typer

from typing import Optional
from tqdm.auto import tqdm
from src.controllers.ScenarioController import ScenarioController

from src.logger import get_logger
from src.dataset_creation.utils import load_queries, get_config, generate_responses
from src.cli.utils import is_api_running

LOGGER = get_logger(__name__)


def create_generations(
    query_file: str,
    output_file: str,
    config_path: str = "src/dataset_creation/configs/base_config.yaml",
    batch_size: int = 30,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    unfiltered: bool = False,
):
    """
    Generate responses for a set of queries and save them to a file.

    The API must be running before running this CLI.

    :param str query_file: csv/jsonl file containing queries and optionally document IDs
    :param str output_file: jsonl path to save the generations to
    :param str config_path: path to config yaml
    :param Optional[int] limit: optionally limit the number of queries to process,
        defaults to None.
    :param Optional[int] offset: optionally skip the first N queries, defaults to None.
    :param int batch_size: number of queries to process in each batch, defaults to 30.
    :param bool unfiltered: whether to use the models in unfiltered generation mode, defaults to False.
    """
    if not is_api_running():
        raise Exception(
            "API is not running. Please start the API before running this script."
        )

    config = get_config(config_path)
    documents_per_query = config["documents_per_query"]
    queries = load_queries(query_file, documents_per_query)

    if limit:
        queries = queries.head(limit)

    if offset:
        LOGGER.info(f"Skipping the first {offset} queries.")
        queries = queries.iloc[offset:]

    queries_batched = [
        queries[i : i + batch_size] for i in range(0, len(queries), batch_size)
    ]

    if output_file.startswith("s3://"):
        LOGGER.warning(
            "Batching is not supported when writing to S3. Ignoring batch_size."
        )
        generations = generate_responses(queries, config, unfiltered)
        generations.to_json(output_file, orient="records", lines=True, index=True)

    else:
        for query_batch in tqdm(
            queries_batched,
            desc="Processing batches",
            total=len(queries_batched),
            unit="batch",
        ):
            generations = generate_responses(query_batch, config, unfiltered)

            with open(output_file, "a") as f:
                generations.to_json(f, orient="records", lines=True, index=True)


if __name__ == "__main__":
    typer.run(create_generations)
