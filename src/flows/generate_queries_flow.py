import os
from prefect import flow, get_run_logger, task
import psutil
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.get_db import get_db
from src.flows.tasks.data_tasks import create_queries, show_db_stats
from cpr_data_access.models import BaseDocument
from prefect.utilities.annotations import quote
from prefect.tasks import exponential_backoff
import argparse

from peewee import Database
from src.flows.utils import get_labs_session
from src.flows.tasks.s3_tasks import get_doc_ids_from_s3, get_file_from_s3

# TODO PR this into CPR SDK to allow session to be passed in


@task(log_prints=True)
def spawn_query_tasks(doc_ids: list[str], tag: str, config: str):
    for doc_id in doc_ids:
        generate_queries_for_document.submit(doc_id, tag, config, get_db())


@task(
    task_run_name="generate_queries_{doc_id}_{tag}",
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_llm"],
)
def generate_queries_for_document(
    doc_id: str,
    tag: str,
    config: str,
    db: Database,
    s3_prefix: str = "project-rag/data/cpr_embeddings_output",
):
    logger = get_run_logger()
    get_labs_session(set_as_default=True)

    logger.info("Initializing scenario controller")
    sc = ScenarioController().from_config(config)

    logger.info("Loading seed queries")
    seed_queries = sc.load_seed_queries()

    logger.info(f"Loading document from s3: {s3_prefix}/{doc_id}")
    s3_bucket = s3_prefix.split("/")[0]
    s3_prefix_path = "/".join(s3_prefix.split("/")[1:])

    doc_json = get_file_from_s3(s3_bucket, f"{s3_prefix_path}/{doc_id}.json")
    with open(f"./data/doc_cache/{doc_id}.json", "w") as f:
        f.write(doc_json)

    doc = BaseDocument.load_from_local("./data/doc_cache/", doc_id)

    logger.info("Initializing RAG controller")
    rc = RagController()

    show_db_stats(db)
    for scenario in sc:
        try:
            logger.info(f"Generating queries for {doc.document_id} with tag {tag}")

            queries = rc.generate_queries(
                document=doc, scenario=scenario, seed_queries=seed_queries, tag=tag
            )

            logger.info(
                f"Created {len(queries)} queries for {scenario.prompt.prompt_template}"
            )
        except Exception as e:
            logger.error(f"Error generating queries for {doc.document_id}: {e}")
            raise e

        logger.info(f"Created {len(queries)} queries")
        create_queries(quote(queries), quote(db))  # type: ignore

    show_db_stats(db)


@flow(log_prints=True)
def query_control_flow(
    config: str = "src/configs/experiment_MAIN_QUERIES_1.0.yaml",
    limit: int = 100,
    offset: int = 0,
    tag: str = "main_run_21_08_2024_queries",
):
    logger = get_run_logger()
    logger.info(
        f"🚀 Starting query control flow with config: {config}, limit: {limit}, offset: {offset}, tag: {tag}"
    )
    logger.info(
        f"Memory before task: {psutil.Process(os.getpid()).memory_info()[0] / float(1024 * 1024)}MiB"
    )

    doc_ids = get_doc_ids_from_s3()

    spawn_query_tasks(doc_ids[offset : min(offset + limit, len(doc_ids))], tag, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping queries together")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiment_MAIN_QUERIES_1.0.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limit the number of documents to process",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset for the documents to process"
    )
    args = parser.parse_args()

    query_control_flow(
        config=args.config, tag=args.tag, limit=args.limit, offset=args.offset
    )
    # generate_queries_for_document(args.doc_id, args.tag, args.config, get_db())
