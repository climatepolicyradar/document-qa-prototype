import argparse
from prefect import flow, get_run_logger, task
from src.controllers.EvaluationController import EvaluationController

from src.flows.tasks.data_tasks import get_answers_needing_evals
from src.models.data_models import QAPair


@flow
def generate_evals_flow(tag: str, limit: int = 5):
    logger = get_run_logger()

    answers = get_answers_needing_evals(tag, limit=limit)
    logger.info(f"💡 Generating evals for {len(answers)} answers with tag {tag}")

    for answer in answers:
        evaluate_answer.submit(answer)


@task
def evaluate_answer(answer: QAPair):
    """Evaluate a single answer from the DB"""
    logger = get_run_logger()
    ec = EvaluationController()

    gen = answer.to_end_to_end_generation()
    logger.info(f"📋 {gen.rag_request.query}: {gen.get_answer()}")

    result = ec.evaluate_all(gen)
    logger.info(f"📋 Result: {result}")

    for score in result:
        answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()

    logger.info(f"📋 Evaluations: {answer.evals}")
    answer.save()


def evaluate_system_response(answer: QAPair):
    """Evaluate a single answer from the DB"""
    logger = get_run_logger()
    ec = EvaluationController()

    gen = answer.to_end_to_end_generation()
    logger.info(f"📋 {gen.rag_request.query}: {gen.get_answer()}")

    score = ec.evaluate_system_response(gen)
    logger.info(f"📋 Result: {score}")
    answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()
    answer.save()
    logger.info(f"📋 Evaluations: {answer.evals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evals for answers")
    parser.add_argument("tag", type=str, help="Tag for grouping QA pairs together")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/answer_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of answers to generate evals for"
    )
    args = parser.parse_args()

    generate_evals_flow(args.tag, args.limit)
