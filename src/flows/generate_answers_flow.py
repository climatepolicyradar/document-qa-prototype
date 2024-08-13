import argparse
import time
from prefect import flow, get_run_logger, task
from src.controllers.DocumentController import DocumentController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.qa_tasks import generate_answer_task

from peewee import Database
from src.flows.utils import get_db
from src.models.data_models import Query, Scenario
from src.flows.tasks.data_tasks import get_queries, save_answer, get_unanswered_queries


@task
def spawn_answer_tasks(
    scenario: Scenario, db: Database, tag: str, query_tag: str, only_new: bool
):
    logger = get_run_logger()

    if only_new:
        # Filter only to the queries in the current set. This is a very poor way to structure this. Just wanna get it done for now. Refactor later.
        query_prompts = [
            "evals-0.0.1/queries-bias",
            "evals-0.0.1/queries-ambiguous",
            "evals-0.0.1/queries-jailbreak",
            "evals-0.0.1/queries-typo",
            "evals-0.0.1/queries-normal",
            "evals-0.0.1/queries-nonsense",
            "evals-0.0.1/queries-factual-errors",
            "evals-0.0.1/queries-opinions",
            "evals-0.0.1/queries-harmful",
            "evals-0.0.1/queries-ambiguous",
        ]

        queries = get_unanswered_queries(
            db,
            tag,
            query_tag,
            scenario.model,
            scenario.prompt.prompt_template,
            query_prompts,
        )
    else:
        queries = get_queries(db, query_tag)

    logger.info(
        f"💡 Generating answers for {len(queries)} queries with tag {query_tag}"
    )
    for i, query in enumerate(queries):
        generate_answer_full.submit(query, scenario, db, tag, query_tag)
        logger.info(f"🕐 Sleeping for 5 seconds before processing query: {query}")
        time.sleep(5)


@task
def generate_answer_full(
    query: Query, scenario: Scenario, db: Database, tag: str, query_tag: str
):
    dc = DocumentController()
    rc = RagController()

    logger = get_run_logger()
    logger.info(f"📋 DB: {db}")

    assert query.document_id is not None, "Document ID is None"
    scenario.document = dc.create_base_document(str(query.document_id))

    logger.info(f"📋 Scenario: {scenario}")
    logger.info(f"💡 Generating answer for query: {query}")
    result = generate_answer_task(query, scenario, tag, rc)

    # Save to database
    save_answer(tag, result, db, query)


@flow
def answer_control_flow(
    config: str,
    tag: str,
    query_tag: str,
    only_new: bool = True,
):
    """
    Flow for generating answers for queries.

    If only_new is True, it will only generate answers for new queries that don't have answers yet for the given model/prompt/tag combination.
    """
    sc = ScenarioController.from_config(config)
    db = get_db()

    for scenario in sc.scenarios:
        spawn_answer_tasks.submit(scenario, db, tag, query_tag, only_new).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping outputs together")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/answer_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--only_new",
        type=bool,
        default=True,
        help="Only generate answers for new queries",
    )
    parser.add_argument(
        "--query_tag", type=str, help="Tag for selecting grouped queries to answer"
    )
    args = parser.parse_args()

    answer_control_flow(args.config, args.tag, args.query_tag, args.only_new)
