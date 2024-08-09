from prefect import flow, get_run_logger
from src.flows.generate_queries_flow import get_doc_ids_from_s3, spawn_query_flows, generate_queries_for_document

@flow(log_prints=True)
def e2e_flow(tag: str, limit: int = 0, offset: int = 0):
    logger = get_run_logger()
    
    query_config = "src/configs/eval_prefect_1_queries_config.yaml"
    answer_config = "src/configs/eval_prefect_1_answers_config.yaml"

    doc_ids = get_doc_ids_from_s3()
    
    if limit == 0:
        limit = len(doc_ids)
    
    doc_ids = doc_ids[offset:limit]
    
    spawn_query_flows(doc_ids, tag, query_config)
    
    
if __name__ == "__main__":
    e2e_flow("test-e2e-prefect-1", limit=2, offset=0)