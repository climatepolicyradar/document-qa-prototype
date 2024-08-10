import os
import time
from prefect import flow, get_run_logger, task
import psutil
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.data_tasks import create_queries, show_db_stats
from cpr_data_access.models import BaseDocument
from prefect.utilities.annotations import quote
from prefect.tasks import exponential_backoff
import argparse
import boto3

from peewee import Database
from src.flows.utils import get_db, get_labs_session

# TODO PR this into CPR SDK to allow session to be passed in


def get_json_filenames(
    bucket_name: str,
    directory_path: str = "",
    session: boto3.Session = get_labs_session(),
) -> list[str]:
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    json_filenames = []

    # Ensure the directory path ends with a '/' if it's not empty
    if directory_path and not directory_path.endswith("/"):
        directory_path += "/"

    try:
        # Paginate through the objects in the bucket
        for page in paginator.paginate(Bucket=bucket_name, Prefix=directory_path):
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Check if the file has a .json extension
                    if obj["Key"].lower().endswith(".json"):
                        json_filenames.append(obj["Key"])

        return json_filenames
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


@task
def get_doc_ids_from_s3(
    bucket_name: str = "project-rag", prefix: str = "data/cpr_embeddings_output"
) -> list[str]:
    logger = get_run_logger()
    logger.info(f"🚀 Getting doc ids from s3 with prefix: {prefix}")
    logger.info(
        f"Memory before task: {psutil.Process(os.getpid()).memory_info()[0] / float(1024 * 1024)}MiB"
    )

    prefixes = get_json_filenames(bucket_name, prefix, get_labs_session())
    logger.info(f"🚀 Got {len(prefixes)} prefixes")

    doc_ids = [file.split("/")[-1].rstrip(".json") for file in prefixes]

    logger.info(f"🚀 Got {len(doc_ids)} doc ids")
    return doc_ids


@task(log_prints=True)
def spawn_query_tasks(doc_ids: list[str], tag: str, config: str):
    logger = get_run_logger()
    for doc_id in doc_ids:
        logger.info(f"🕐 Sleeping for 5 seconds before processing doc_id: {doc_id}")
        time.sleep(5)
        generate_queries_for_document.submit(doc_id, tag, config, get_db())


#
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
    doc = BaseDocument.load_from_remote(s3_prefix, doc_id)

    logger.info("Initializing RAG controller")
    rc = RagController()

    show_db_stats(db)
    for scenario in sc:
        try:
            logger.info(f"Generating queries for {doc.document_id} with tag {tag}")

            queries = rc.generate_queries(
                document=doc, scenario=scenario, seed_queries=seed_queries, tag=tag
            )

        except Exception as e:
            logger.error(f"Error generating queries for {doc.document_id}: {e}")
            raise e

        logger.info(f"Created {len(queries)} queries")
        create_queries(quote(queries), quote(db))  # type: ignore

    show_db_stats(db)


@flow(log_prints=True)
def query_control_flow(
    config: str = "src/configs/eval_prefect_1_queries_config.yaml",
    limit: int = 100,
    offset: int = 0,
    tag: str = "test",
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
        default="src/configs/eval_prefect_1_queries_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--doc_id",
        type=str,
        default="CCLW.executive.4840.1833",
        help="Optionally set a document id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
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
