import argparse
from prefect import flow, get_run_logger, task
from src.controllers.EvaluationController import EvaluationController
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
    get_query_by_id,
)
from src.flows.queue import get_queue_job, queue_job, mark_job_done


@task
def queue_all_answer_tasks(scenarios: list[Scenario], tag: str, query_tag: str):
    db = get_db()
    logger = get_run_logger()

    queries = get_queries(db, query_tag)

    # We do it this way to invert scenario order so that we can queue across model,prompt combinations and process them in parallel rather than processing all queries for a given model/prompt combination sequentially
    offset = 0
    num_per_loop = 50
    while offset < len(queries):
        for scenario in scenarios:
            for i, query in enumerate(queries[offset : offset + num_per_loop]):
                logger.info(
                    f"📋 Queueing job {i} of {num_per_loop} (offset {offset} of {len(queries)} total)"
                )
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

        # Once we've done all the scenarios, we move to the next chunk
        offset += num_per_loop


@flow
def process_answer_job_from_queue(
    tag: str = "main_experiment_run_2024_08_26", limit: int = 15
):
    db = get_db()
    logger = get_run_logger()
    dc = DocumentController()
    rc = RagController()
    ec = EvaluationController()
    ec.set_evaluators(
        [
            "formatting",
            "g_eval_policy",
            "g_eval_faithfulness_llama3",
            "patronus_lynx",
            "vectara",
        ]
    )

    for i in range(limit):
        job = get_queue_job(tag)
        if job is None:
            logger.info("📋 Could not get job from queue")
            break

        logger.info(f"📋 Job: {job}")

        if job["generation_engine"] == "openai":
            logger.warning(
                f"Skipping job {job['query_id']} because generation engine is openai"
            )
            continue

        scenario = Scenario(
            model=job["model"],
            prompt=Prompt.from_template(job["prompt"]),
            generation_engine=job["generation_engine"],
            src_config=job["src_config"],
            document=dc.create_base_document(job["document_id"]),
        )

        query = get_query_by_id(db, job["query_id"])

        generate_answer_full(query, scenario, db, tag, job["query_tag"], rc, ec)

        mark_job_done(tag, job["receipt_handle"])


def generate_answer_full(
    query: Query,
    scenario: Scenario,
    db: Database,
    tag: str,
    query_tag: str,
    rc: RagController,
    ec: EvaluationController,
):
    logger = get_run_logger()
    dc = DocumentController()

    assert query.document_id is not None, "Document ID is None"
    scenario.document = dc.create_base_document(str(query.document_id))

    logger.info(f"📋 Scenario: {scenario}")
    logger.info(f"💡 Generating answer for query: {query}")
    result = generate_answer_task(query, scenario, tag, rc)

    # Save to database
    answer = save_answer(tag, result, db, query)
    if not result.rag_response.refused_answer():  # type: ignore
        try:
            result = ec.evaluate_all(result)

            for score in result:
                answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()

            logger.info(f"📋 Evaluations: {answer.evals}")
            answer.save()
        except Exception as e:
            logger.error(f"🚨 Error evaluating answer: {e}")

    return result


@flow
def queue_answer_flow(config: str, tag: str, query_tag: str):
    """
    Flow for generating answers for queries.

    If only_new is True, it will only generate answers for new queries that don't have answers yet for the given model/prompt/tag combination.
    """
    sc = ScenarioController.from_config(config)

    queue_all_answer_tasks(sc.scenarios, tag, query_tag)


def main(tag: str, config: str, query_tag: str):
    process_answer_job_from_queue(tag, limit=15)
    #
    # spawn_answer_tasks(tag, limit=2)
    # queue_answer_flow(config, tag, query_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping outputs together")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiment_MAIN_ANSWERS_1.0.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--query_tag",
        type=str,
        help="Tag for selecting grouped queries to answer",
        default="",
    )
    args = parser.parse_args()
    main(args.tag, args.config, args.query_tag)
