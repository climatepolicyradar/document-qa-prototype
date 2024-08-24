from prefect import flow, get_run_logger
from src.controllers.DocumentController import DocumentController
from src.controllers.RagController import RagController
from src.controllers.EvaluationController import EvaluationController
from src.flows.queue import get_queue, queue_job

from src.flows.tasks.data_tasks import get_query_by_id, save_answer
from src.flows.tasks.qa_tasks import generate_answer_task
from src.flows.utils import get_db
from src.models.data_models import Prompt, QAPair, Scenario
from peewee import fn


@flow
def actually_well_setup_evals_experiments(
    tag: str = "g_eval_comparison_experiment_2", limit: int = 15
):
    db = get_db()
    logger = get_run_logger()

    cursor = db.execute_sql(
        """    
    SELECT * 
    FROM (
            SELECT 
                qa.id,
                qa.generation,
                qa.question,
                qa.answer,
                qa.evals,
                qa.model,
                qa.prompt,
                qa.updated_at,
                qa.generation,
                q.prompt,
                ROW_NUMBER() OVER(PARTITION BY qa.model, qa.prompt, q.prompt) AS rn
            FROM qapair qa
                JOIN dbquery q ON qa.query_id = q.id
            WHERE pipeline_id = 'main_answer_run_2024_08_10' 
                AND qa.model NOT IN ('neural-chat-7b')
                AND qa.prompt NOT IN ('evals-answers-0.0.1/explain_assumptions', 'evals-answers-0.0.1/branch', 'evals-answers-0.0.1/kg_intermediate')
                AND q.prompt NOT IN ('evals-0.0.1/queries-harmful', 'evals-0.0.1/queries-inference', 'evals-0.0.1/queries-esl', 'evals-0.0.1/queries-complex', 'evals-0.0.1/queries-bias', 'evals-0.0.1/queries-factual-errors', 'evals-0.0.1/queries-opinions', 'evals-0.0.1/queries-jailbreak', 'evals-0.0.1/queries-nonsense')
        ) a
    WHERE rn < 750;"""
    )

    count = 0
    for row in cursor.fetchall():
        run_this_row = False
        """
        We want to ensure that evals has:
        
        g_eval_faithfulness_gpt4o
        g_eval_faithfulness_gemini
        g_eval_faithfulness_llama3
        patronus_lynx
        vectara
        
        If it doesn't have all of these, we need to evaluate it
        In which case we'll push it onto the queue
        """
        evals = row[4]
        eval_keys = [
            "g_eval-faithfulness_gpt4o",
            "g_eval-faithfulness_gemini",
            "g_eval-faithfulness_llama3",
            "patronus_lynx-faithfulness",
            "vectara-faithfulness",
        ]

        if evals is None or len(evals) == 0:
            run_this_row = True
        else:
            for e in eval_keys:
                if e not in evals:
                    logger.info(f"📋 Row {row[0]} missing eval {e}")
                    run_this_row = True
                    break

        if run_this_row:
            count += 1
            logger.info(f"📋 Row run: {row[0]} ({count})")

            # We only need the ID for the job because the consumer will get the rest of the data from the DB -- no scenario info needed its baked into the evals.
            queue_job(tag, row[0])


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
        .where(QAPair.evals.has_key("g_eval_faithfulness_gpt4o") == False)  # noqa: E712
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
    actually_well_setup_evals_experiments()
