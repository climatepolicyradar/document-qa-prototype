import argparse
from prefect import flow, get_run_logger
from src.controllers.EvaluationController import EvaluationController

from peewee import Database
from src.flows.utils import get_db
from src.flows.tasks.data_tasks import get_answers_needing_evals


@flow
def generate_evals_flow(db: Database, tag: str, limit: int = 5):
    logger = get_run_logger()

    ec = EvaluationController()

    answers = get_answers_needing_evals(db, tag, limit=limit)
    while len(answers) > 0:
        logger.info(f"💡 Generating evals for {len(answers)} answers with tag {tag}")

        for answer in answers:
            gen = answer.to_end_to_end_generation()
            logger.info(f"📋 {gen.rag_request.query}: {gen.get_answer()}")

            result = ec.evaluate_all(gen)
            logger.info(f"📋 Result: {result}")

            for score in result:
                answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()

            print(answer.evals)
            answer.save()

        logger.info("🔄 Getting more answers to evaluate")
        answers = get_answers_needing_evals(db, tag, limit=limit)


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

    db = get_db()

    generate_evals_flow(db, args.tag, args.limit)
