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
from src.flows.tasks.data_tasks import get_queries, save_answer

@flow
def generate_answers_flow(
    scenario: Scenario,
    db: Database,
    tag: str
):
    dc = DocumentController()
    rc = RagController()
    
    # TODO how do we want to specify which queries get answered?
    logger = get_run_logger()
    logger.info(f"📋 DB: {db}")
    
    queries = get_queries(db, tag)
    logger.info(f"💡 Generating answers for {len(queries)} queries with tag {tag}")
    
    for i,query in enumerate(queries):
        scenario.document = dc.create_base_document(query.document_id)
                
        logger.info(f"📋 Scenario: {scenario}")
        logger.info(f"💡 Generating answer for query: {query}")
        result = generate_answer_task.submit(query, scenario, tag, rc)
        
        #Save to database
        save_answer.submit(tag, result, db)
        
        # Because the concurrency can get a bit nuts 
        if i % 5 == 0 and i != 0:
            logger.info("⏳ Waiting for 5 seconds...")
            import time
            time.sleep(5)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping queries together")
    parser.add_argument("--config", type=str, default="src/configs/answer_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    sc = ScenarioController.from_config(args.config)
    
    db = get_db()
    
    for scenario in sc.scenarios:
        generate_answers_flow(scenario, db, args.tag)