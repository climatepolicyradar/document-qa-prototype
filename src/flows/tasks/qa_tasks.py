from prefect import task, get_run_logger
from src.controllers.EvaluationController import EvaluationController
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import Scenario, ScenarioController
from src.models.data_models import EndToEndGeneration, Query
from cpr_data_access.models import BaseDocument
from prefect.tasks import exponential_backoff
import jinja2

@task(
    task_run_name="create_queries_{document.document_id}_{tag}",
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_llm"]
)
def create_query_list_task(
    scenario: Scenario,
    document: BaseDocument,
    seed_queries: list[Query],
    sc: ScenarioController,
    tag: str
) -> list[Query]:
    logger = get_run_logger()
    
    rc = RagController()
    
    try:
        logger.info(f"Generating queries for {document.document_id} with tag {tag}")
        response = rc.generate_queries(
            document=document,
            scenario=scenario,
            seed_queries=seed_queries,
            tag=tag
        )
        
    except Exception as e:
        logger.error(
                    f"Error generating queries for {document.document_id}: {e}"
        )
        raise e
    
    
    return response


@task(
    task_run_name="generate_answers_{tag}",
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=120,
    tags=["calls_llm"]
)
def generate_answer_task(
    query: Query,
    scenario: Scenario,
    tag: str,
    rc: RagController
) -> EndToEndGeneration:
    logger = get_run_logger()
    
    logger.info(f"💡 Generating answer for {query.uuid}: \"{query.text}\"")
    try:
        response = rc.run_rag_pipeline(
            query=query.text,
            scenario=scenario
        )
    except Exception as e:
        logger.error(
            f"🚨 Error generating answer for {query.uuid}: \"{query.text}\": {e}"
        )
        raise e
    
    logger.info(f"💡 Answer: {response.rag_response.text}")
    return response


@task(
    task_run_name="generate_evals_{generation.rag_request.document_id}_{tag}",
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_llm"]
)
def run_eval_task(
    generation: EndToEndGeneration,
    eval: str, 
    kwargs: dict
) -> str:
    logger = get_run_logger()
    
    ec = EvaluationController()
    
    try:
        result = ec.evaluate(
            generation=generation,
            eval=eval,
            eval_kwargs=kwargs
        )
    except Exception as e:
        logger.error(f"🚨 Error evaluating answer for {generation.uuid} with {eval}: {e}")
        raise e
    
    return result