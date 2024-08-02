from prefect import flow, get_run_logger
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.qa_tasks import create_query_list_task
from src.flows.tasks.data_tasks import create_queries, show_db_stats
from cpr_data_access.models import Dataset, BaseDocument
import argparse

from peewee import Database
from src.flows.utils import get_db

@flow(log_prints=True)
def generate_queries_flow(
    sc: ScenarioController,
    docs: Dataset,
    db: Database,
    tag: str,
    limit: int | None = None,
    offset: int | None = None
):
    logger = get_run_logger()
    
    seed_queries = sc.load_seed_queries()
    
    if limit is None:
        limit = len(docs.documents)
    
    if offset is None:
        offset = 0
    
    show_db_stats(db)
    for document in docs.documents[offset:limit]:      
        for scenario in sc:
            try:
                queries = create_query_list_task.submit(
                    scenario=scenario,
                    document=document, 
                    seed_queries=seed_queries,
                    sc=sc,
                    tag=tag
                ) # type: ignore
                create_queries.submit(queries, db)
                
            except Exception as e:
                logger.error(f"Error generating queries: {e}")
            
    
    show_db_stats(db)
    

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for documents")
    parser.add_argument("tag", type=str, help="Tag for grouping queries together")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/configs/query_config.yaml", 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Optionally limit the number of documents to process"
    )
    parser.add_argument(
        "--offset", 
        type=int, 
        default=None, 
        help="Optionally skip the first N documents"
    )
    args = parser.parse_args()
    
    sc = ScenarioController().from_config(args.config)
    
    documents = ( Dataset(BaseDocument).load_from_remote("s3://project-rag/data/documents_all/").filter_by_language("en") )
    
    generate_queries_flow(sc, documents, get_db(), args.tag, args.limit, args.offset)