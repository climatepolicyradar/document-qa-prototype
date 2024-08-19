import argparse
from prefect import flow, get_run_logger, task
from src.controllers.DocumentController import DocumentController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.qa_tasks import generate_answer_task
from peewee import Database
from src.flows.utils import get_db
from src.models.data_models import Prompt, Query, Scenario
from src.flows.tasks.data_tasks import (
    get_queries,
    save_answer,
    get_unanswered_queries,
    get_query_by_id,
)
from src.flows.generate_evals_flow import evaluate_system_response
from src.flows.queue import queue_job, get_queue


@task
def queue_answer_tasks(
    scenarios: list[Scenario], tag: str, query_tag: str, only_new: bool
):
    db = get_db()
    logger = get_run_logger()

    continue_queueing = True
    done_scenarios = {}

    # We do it this way to invert scenario order so that we can queue across model,prompt combinations and process them in parallel rather than processing all queries for a given model/prompt combination sequentially
    while continue_queueing:
        for scenario in scenarios:
            scenario_key = f"{scenario.model}-{scenario.prompt.prompt_template}"
            if scenario_key in done_scenarios:
                continue

            if only_new:
                queries = get_unanswered_queries(
                    db, tag, query_tag, scenario.model, scenario.prompt.prompt_template
                )
            else:
                queries = get_queries(db, query_tag)

            limit = min(len(queries), 10)

            if limit < 10:
                logger.info(f"📋 No more queries to queue for scenario {scenario}")
                done_scenarios[scenario_key] = True

            if len(done_scenarios) == len(scenarios):
                continue_queueing = False

            for i, query in enumerate(queries[:limit]):
                logger.info(f"📋 Queueing job {i} of {limit} ({len(queries)} total)")
                queue_job(
                    tag,
                    {
                        "query_id": query.db_id,
                        "model": scenario.model,
                        "prompt": scenario.prompt.prompt_template,
                        "generation_engine": scenario.generation_engine,
                        "document_id": query.document_id,
                        "src_config": scenario.src_config,
                        "tag": tag,
                        "query_tag": query_tag,
                    },
                )


@flow
def run_answers_from_queue(tag: str = "prompt-answer-experiment", limit: int = 5):
    db = get_db()
    logger = get_run_logger()
    dc = DocumentController()

    q = get_queue(tag)

    for i in range(limit):
        job = q.get()
        logger.info(f"📋 Job: {job}")
        logger.info(f"📋 Job scenario: {job.data}")

        scenario = Scenario(
            model=job.data["model"],
            prompt=Prompt.from_template(job.data["prompt"]),
            generation_engine=job.data["generation_engine"],
            src_config=job.data["src_config"],
            document=dc.create_base_document(job.data["document_id"]),
        )

        query = get_query_by_id(db, job.data["query_id"])

        generate_answer_full.submit(query, scenario, db, tag, job.data["query_tag"])


@task(tags=["generate_answer"])
def generate_answer_full(
    query: Query, scenario: Scenario, db: Database, tag: str, query_tag: str
):
    dc = DocumentController()
    rc = RagController()

    logger = get_run_logger()

    assert query.document_id is not None, "Document ID is None"
    scenario.document = dc.create_base_document(str(query.document_id))

    logger.info(f"📋 Scenario: {scenario}")
    logger.info(f"💡 Generating answer for query: {query}")
    result = generate_answer_task(query, scenario, tag, rc)

    # Save to database
    qa_pair = save_answer(tag, result, db, query)

    evaluate_system_response(qa_pair)


@flow
def queue_answer_flow(
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

    queue_answer_tasks(sc.scenarios, tag, query_tag, only_new)


def main(tag: str, config: str, query_tag: str, only_new: bool):
    # run_answers_from_queue(tag, limit=2)
    # spawn_answer_tasks(tag, limit=2)
    queue_answer_flow(config, tag, query_tag, only_new)


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
        "--query_tag",
        type=str,
        help="Tag for selecting grouped queries to answer",
        default="",
    )
    args = parser.parse_args()
    main(args.tag, args.config, args.query_tag, args.only_new)
