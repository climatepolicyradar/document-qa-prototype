import argparse
from prefect import flow, get_run_logger
from src.controllers.DocumentController import DocumentController
from src.controllers.RagController import RagController
from src.controllers.EvaluationController import EvaluationController
from src.flows.queue import get_queue

from src.flows.tasks.data_tasks import get_query_by_id, save_answer
from src.flows.tasks.qa_tasks import generate_answer_task
from src.flows.utils import get_db
from src.models.data_models import Prompt, QAPair, Scenario
from peewee import fn


@flow
def create_gpt4_evals_flow(tag: str = "g_eval_comparison_experiment", limit: int = 15):
    logger = get_run_logger()
    ec = EvaluationController()
    ec.set_evaluators(
        [
            "g_eval_faithfulness_gpt4o",
        ]
    )

    # Get answers for this run with no gpt-4o eval
    answers = (
        QAPair.select()
        .where(QAPair.evals.has_key("g_eval_faithfulness_gpt4o") is False)
        .where(QAPair.pipeline_id == tag)
        .order_by(fn.Random())
        .limit(limit)
    )

    logger.info(f"🎲 Got {len(answers)} answers with tag {tag} with no gpt-4o eval")

    for answer in answers:
        logger.info(f"🎲 Evaluating answer {answer.id}")
        result = ec.evaluate_all(answer.to_end_to_end_generation())
        logger.info(f"🎲 Result: {result}")

        for score in result:
            answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()

        logger.info(f"📋 Evaluations: {answer.evals}")
        answer.save()


@flow
def process_faithfulness_experiment_answer_job(
    tag: str = "g_eval_comparison_experiment", limit: int = 15
):
    db = get_db()
    logger = get_run_logger()
    dc = DocumentController()
    rc = RagController()
    ec = EvaluationController()
    ec.set_evaluators(
        [
            "system_response",
            "g_eval_faithfulness_gemini",
            "g_eval_faithfulness_llama3",
            "patronus_lynx",
            "vectara",
        ]
    )

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

        logger.info(f"📋 Scenario: {scenario}")
        logger.info(f"💡 Generating answer for query: {query}")

        result = generate_answer_task(query, scenario, tag, rc)
        # Save to database
        qa_pair = save_answer(tag, result, db, query)
        if result.rag_response is None or result.rag_response.refused_answer():
            logger.warning(
                "RAG response is None or refused answer, skipping evaluation"
            )
            continue

        result = ec.evaluate_all(result)
        logger.info(f"📋 Result: {result}")

        for score in result:
            qa_pair.evals[f"{score.name}-{score.type}"] = score.model_dump_json()

        logger.info(f"📋 Evaluations: {qa_pair.evals}")
        qa_pair.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evals for answers")
    parser.add_argument("tag", type=str, help="Tag for grouping QA pairs together")

    args = parser.parse_args()

    create_gpt4_evals_flow(args.tag)
