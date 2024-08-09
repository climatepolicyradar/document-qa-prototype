from prefect import flow, get_run_logger
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController
from src.flows.tasks.qa_tasks import create_query_list_task
from src.flows.tasks.data_tasks import create_queries, show_db_stats
from cpr_data_access.models import Dataset, BaseDocument
from cpr_data_access.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
import argparse

from peewee import Database
from src.flows.utils import get_db

@flow(log_prints=True)
def get_doc_ids_from_s3(prefix: str = "s3://project-rag/data/cpr_embeddings_output") -> list[str]:
    prefixes = _get_s3_keys_with_prefix(prefix)
    json_files = [prefix for prefix in prefixes if prefix.endswith('.json')]
    doc_ids = [file.split('/')[-1].rstrip('.json') for file in json_files]
    return doc_ids

@flow(log_prints=True)
def spawn_query_flows(doc_ids: list[str], tag: str, config: str):
    for doc_id in doc_ids:
        generate_queries_for_document(doc_id, tag, config, get_db())

#
@flow(log_prints=True)
def generate_queries_for_document(
    doc_id: str,
    tag: str,
    config: str,
    db: Database,
    s3_prefix: str = "project-rag/data/cpr_embeddings_output"
):
    logger = get_run_logger()
    
    sc = ScenarioController().from_config(config)
    seed_queries = sc.load_seed_queries()

    doc = BaseDocument.load_from_remote(s3_prefix, doc_id)
    
    rc = RagController(observe=False)
    
    show_db_stats(db)
    for scenario in sc:
        try:
            queries = create_query_list_task(
                scenario=scenario,
                document=doc, 
                seed_queries=seed_queries,
                rc=rc,
                tag=tag
            ) # type: ignore
            create_queries(queries, db)
            
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
        "--doc_id", 
        type=str, 
        default="CCLW.executive.4840.1833", 
        help="Optionally set a document id"
    )
    args = parser.parse_args()
    
    doc_ids = get_doc_ids_from_s3()
    spawn_query_flows(doc_ids[:5], args.tag, args.config)
    #generate_queries_for_document(args.doc_id, args.tag, args.config, get_db())