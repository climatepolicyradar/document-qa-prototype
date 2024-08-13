import argparse
from prefect import flow, get_run_logger
from src.controllers.EvaluationController import EvaluationController

from peewee import Database
from src.flows.utils import get_db
from src.flows.tasks.data_tasks import get_qa_pairs_with_evals


@flow
def generate_analysis_flow(db: Database, tag: str, limit: int = 5):
    """Generates the analysis of a set of evaluations"""
    logger = get_run_logger()

    qa_pairs = get_qa_pairs_with_evals(db, tag, limit=limit)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis for the evaulation set")
    parser.add_argument("tag", type=str, help="Tag for grouping QA pairs together")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of answers to generate evals for"
    )
    args = parser.parse_args()

    db = get_db()

    generate_analysis_flow(db, args.tag, args.limit)
