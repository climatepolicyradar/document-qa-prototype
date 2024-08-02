import argparse
from prefect import flow, get_run_logger
from src.controllers.DocumentController import DocumentController
from src.controllers.EvaluationController import EvaluationController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.qa_tasks import generate_answer_task

from peewee import Database
from src.flows.utils import get_db
from src.models.data_models import Scenario
from src.flows.tasks.data_tasks import get_answers

@flow
def generate_evals_flow(
    db: Database,
    tag: str
):
    logger = get_run_logger()
    
    ec = EvaluationController()
    
    answers = get_answers(db, tag)
    logger.info(f"💡 Generating evals for {len(answers)} answers with tag {tag}")
    
    for answer in answers:
        gen = answer.to_end_to_end_generation()
        logger.info(f"📋 Generation: {gen}")
        
        result = ec.evaluate_all(gen)
        logger.info(f"📋 Result: {result}")
        
        for score in result:
            answer.evals[f"{score.name}-{score.type}"] = score.model_dump_json()
        
        answer.save()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evals for answers")
    parser.add_argument("tag", type=str, help="Tag for grouping QA pairs together")
    parser.add_argument("--config", type=str, default="src/configs/answer_config.yaml", help="Path to the config file")
    args = parser.parse_args()

    
    db = get_db()
    
    generate_evals_flow(db, args.tag)