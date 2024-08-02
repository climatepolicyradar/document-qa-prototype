from pathlib import Path
import random
import time
from typing import Optional

import typer
from tqdm.auto import tqdm
import argilla as rg

from src.dataset_creation.utils import load_queries
from src.models.vespa import get_vespa_app, VespaResponse, process_vespa_response
from src.vespa.query import get_rank_profiles, make_request
from src.logger import get_logger

LOGGER = get_logger(__name__)



def create_retrieval_dataset(
    queries_path: Path,
    output_path: Path,
    sleep_between_queries: float = 0.1,
    random_seed: int = 42,
    limit: Optional[int] = None,
):
    """
    Create a dataset of queries run against Vespa with different rank profiles.

    Saves the dataset as a JSONL file.
    """
    if output_path.exists():
        raise FileExistsError(f"Output path '{output_path}' already exists.")

    if not output_path.suffix == ".jsonl":
        raise ValueError("Output path must be a JSONL file.")

    random.seed(random_seed)
    LOGGER.info(f"Random seed set to {random_seed}")

    vespa_app = get_vespa_app()

    LOGGER.info("Loading queries...")
    queries_df = load_queries(queries_path, documents_per_query=1)
    queries_and_doc_ids = queries_df[["query", "document_id"]].to_dict(orient="records")

    len_before = len(queries_and_doc_ids)
    queries_and_doc_ids = [
        i for i in queries_and_doc_ids if not i["document_id"].isnumeric()
    ]

    n_unique_docs = len(set([i["document_id"] for i in queries_and_doc_ids]))
    n_unique_queries = len(set([i["query"] for i in queries_and_doc_ids]))

    LOGGER.info(
        f"Removed {len_before - len(queries_and_doc_ids)} queries related to UNECE documents."
    )
    LOGGER.info(
        f"Unique queries: {n_unique_queries}, unique documents: {n_unique_docs}"
    )

    if limit:
        queries_and_doc_ids = queries_and_doc_ids[:limit]
        LOGGER.info(f"Limiting to {limit} queries.")

    all_rank_profiles = get_rank_profiles()

    LOGGER.info(
        f"Running {len(queries_and_doc_ids)} queries against Vespa with a sleep of {sleep_between_queries} seconds between each query."
    )

    dataset: list[VespaResponse] = []
    n_failed = 0

    for query_and_doc_id in tqdm(queries_and_doc_ids):
        query = query_and_doc_id["query"]
        doc_id = query_and_doc_id["document_id"]

        rank_profile = random.choice(all_rank_profiles)

        try:
            raw_response = make_request(
                vespa_app, query, doc_id, hits=30, rank_profile=rank_profile
            )
            processed_response = process_vespa_response(raw_response)

            if processed_response is None:
                LOGGER.warning(
                    f"Skipping query '{query}' with document ID '{doc_id}' as it is not in Vespa."
                )
                continue

            response = VespaResponse(
                query=query,
                document_id=doc_id,
                rank_profile=rank_profile,
                results=processed_response,
            )

            dataset.append(response)

        except Exception as e:
            LOGGER.error(
                f"Error processing query '{query}' with document ID '{doc_id}': {e}"
            )
            n_failed += 1

        time.sleep(sleep_between_queries)

    LOGGER.info(f"Finished processing {len(dataset)} queries with {n_failed} failures.")

    output_path.write_text(
        "\n".join([response.model_dump_json() for response in dataset])
    )


if __name__ == "__main__":
    typer.run(create_retrieval_dataset)
