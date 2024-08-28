import os
from prefect import flow, get_run_logger, task
import psutil
from src.controllers.ScenarioController import ScenarioController
from src.controllers.RagController import RagController
from src.flows.tasks.data_tasks import show_db_stats
from prefect.tasks import exponential_backoff
import argparse
import json

from peewee import Database
from src.flows.utils import get_db, get_labs_session
from src.flows.tasks.s3_tasks import (
    get_doc_ids_from_s3,
    get_file_from_s3,
    push_file_to_s3,
)
from src.commands.llm_commands import GetTopicsFromText

# TODO: should this be in the config yaml?
FIRST_N_PAGES = 3
MAX_WORDS_IN_TEXT = 2000


@task(log_prints=True)
def spawn_document_topic_tasks(doc_ids: list[str], tag: str, config: str):
    for doc_id in doc_ids:
        generate_topics_for_document.submit(doc_id, tag, config, get_db())


def get_page_text(document: dict, first_n_pages: int) -> str:
    """Get text from specific pages, or all pages if the document is an HTML document."""

    if document["pdf_data"] is not None:
        blocks = document["pdf_data"]["text_blocks"]
        page_text_list = [
            " ".join(block["text"])
            for block in blocks
            if block["page_number"] < first_n_pages
        ]
    elif document["html_data"] is not None:
        blocks = document["html_data"]["text_blocks"]
        page_text_list = [" ".join(block["text"]) for block in blocks]
    else:
        page_text_list = []

    return " ".join(page_text_list)


def get_first_n_words(text: str, n_words: int) -> str:
    """Return the first max_words of the text."""

    words = text.split()
    if len(words) > n_words:
        return " ".join(words[:n_words])
    else:
        return text


@task(
    task_run_name="generate_document_topics_{doc_id}_{tag}",
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_llm"],
)
def generate_topics_for_document(
    doc_id: str,
    tag: str,
    config: str,
    db: Database,
    s3_prefix: str = "project-rag/data/cpr_embeddings_output_valid_documents",
):
    logger = get_run_logger()
    get_labs_session(set_as_default=True)

    logger.info("Initializing scenario controller")
    sc = ScenarioController().from_config(config)

    logger.info(f"Loading document from s3: {s3_prefix}/{doc_id}")
    s3_bucket = s3_prefix.split("/")[0]
    s3_prefix_path = "/".join(s3_prefix.split("/")[1:])

    doc_json = get_file_from_s3(s3_bucket, f"{s3_prefix_path}/{doc_id}.json")

    doc_dict = json.loads(doc_json)
    document_id = doc_dict["document_id"]

    show_db_stats(db)
    for scenario in sc:
        try:
            logger.info("Initializing RAG controller")
            rc = RagController()
            logger.info(f"Generating topics for {document_id} with tag {tag}")

            first_pages_text = get_page_text(doc_dict, FIRST_N_PAGES)
            first_pages_text = get_first_n_words(first_pages_text, MAX_WORDS_IN_TEXT)

            topics = GetTopicsFromText(rc).process_text(first_pages_text, scenario)

            logger.info(
                f"Created {len(topics)} topics for {scenario.prompt.prompt_template}"
            )
        except Exception as e:
            logger.error(f"Error generating queries for {document_id}: {e}")
            raise e

        output_json = {"document_id": document_id, "topics": topics}

        push_file_to_s3(
            bucket_name=s3_bucket,
            file_path=f"{s3_prefix_path}/{doc_id}_topics.json",
            file_content=json.dumps(output_json),
        )


@flow(log_prints=True)
def document_topic_control_flow(
    config: str,
    limit: int,
    offset: int = 0,
    tag: str = "main_run_28_08_2024_document_topics",
):
    logger = get_run_logger()
    logger.info(
        f"🚀 Starting query control flow with config: {config}, limit: {limit}, offset: {offset}, tag: {tag}"
    )
    logger.info(
        f"Memory before task: {psutil.Process(os.getpid()).memory_info()[0] / float(1024 * 1024)}MiB"
    )

    doc_ids = get_doc_ids_from_s3()

    spawn_document_topic_tasks(
        doc_ids[offset : min(offset + limit, len(doc_ids))], tag, config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping queries together")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiment_document_topics_1.0.yaml",
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

    document_topic_control_flow(
        config=args.config, tag=args.tag, limit=args.limit, offset=args.offset
    )
    # generate_queries_for_document(args.doc_id, args.tag, args.config, get_db())
