from prefect import task, get_run_logger, flow

from peewee import Database

from src.controllers.VespaController import VespaController
from src.models.data_models import EndToEndGeneration, Query, QAPair, DBQuery
from src.flows.utils import get_db
from prefect.tasks import exponential_backoff


def migrate_db(db: Database, drop_tables: bool = False):
    logger = get_run_logger()
    db.connect()
    logger.info("Creating tables...")

    if drop_tables:
        db.drop_tables([DBQuery, QAPair], safe=True)

    db.create_tables([DBQuery, QAPair], safe=True)
    logger.info("Tables created")


@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"],
)
def create_queries(queries: list[Query], db: Database):
    logger = get_run_logger()
    logger.info(f"Persisting {len(queries)} queries...")

    db_models = [DBQuery.from_query(query) for query in queries]
    DBQuery.bulk_create(db_models)


@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"],
)
def get_queries(db: Database, tag: str = "test") -> list[Query]:
    logger = get_run_logger()
    queries = [query.to_query() for query in DBQuery.select().where(DBQuery.tag == tag)]
    logger.info(f"🎲 Got {len(queries)} queries with tag {tag}")
    return queries


def get_unanswered_queries(
    db: Database,
    tag: str,
    query_tag: str,
    model: str,
    prompt: str,
    querstion_prompts: list[str],
) -> list[Query]:
    logger = get_run_logger()
    logger.info(
        f"Getting unanswered queries for tag {tag} with model {model} and prompt {prompt}"
    )
    curr_answer_ids = [
        qa.query_id
        for qa in QAPair.select()
        .where(QAPair.pipeline_id == tag)
        .where(QAPair.model == model)
        .where(QAPair.prompt == prompt)
    ]

    logger.info(
        f"🎲 Got {len(curr_answer_ids)} answers for tag {tag} with model {model} and prompt {prompt}"
    )

    queries = [
        query.to_query()
        for query in DBQuery.select().where(
            DBQuery.tag == query_tag,
            DBQuery.id.not_in(curr_answer_ids),
            DBQuery.prompt.in_(querstion_prompts),
        )
    ]

    logger.info(
        f"🎲 Got {len(queries)} unanswered queries for tag {tag} with model {model} and prompt {prompt}"
    )
    return queries


@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"],
)
def save_answer(tag: str, answer: EndToEndGeneration, db: Database, query: Query):
    logger = get_run_logger()
    logger.info("Saving answer to database...")
    assert query.db_id is not None, "Query database ID is not set!"
    answer.to_db_model(tag, query_id=str(query.db_id)).save()


@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"],
)
def get_answers(db: Database, tag: str) -> list[QAPair]:
    logger = get_run_logger()
    answers = [qa for qa in QAPair.select().where(QAPair.pipeline_id == tag)]
    logger.info(f"🎲 Got {len(answers)} answers")
    return answers


def get_answers_needing_evals(db: Database, tag: str, limit: int = 10) -> list[QAPair]:
    logger = get_run_logger()
    answers = [
        qa
        for qa in QAPair.select()
        .where(QAPair.pipeline_id == tag)
        .where(QAPair.evals == {})
        .limit(limit)
    ]
    logger.info(f"🎲 Got {len(answers)} answers needing evals")
    return answers


def get_qa_pairs_with_evals(db: Database, tag: str, limit: int) -> list[QAPair]:
    """
    Gets the QA pairs with non-empty eval results for a given tag

    TODO: Performs a join with the Query table to get the query prompt type
    """
    logger = get_run_logger()
    answers = [
        qa
        for qa in QAPair.select()
        .where(QAPair.pipeline_id == tag)
        .where(QAPair.evals != {})
        .limit(limit)
    ]
    logger.info(f"🎲 Got {len(answers)} answers with evals")
    return answers


def get_answer_by_id(db: Database, id: int) -> QAPair:
    logger = get_run_logger()
    answer = QAPair.get_or_none(QAPair.id == id)
    logger.info(f"🎲 Got answer with id {id}")
    return answer


@flow
def init_db_and_tables(db: Database, drop_tables: bool = False):
    logger = get_run_logger()
    logger.info("Migrating database...")
    migrate_db(db, drop_tables)
    logger.info("Database migrated")


@flow
def show_db_stats(db: Database):
    logger = get_run_logger()
    logger.info("Getting database stats...")
    logger.info(f"Number of Queries: {DBQuery.select().count()}")
    logger.info(f"Number of QAPairs: {QAPair.select().count()}")
    logger.info(
        f"Number of QAPairs with answer: {QAPair.select().where(QAPair.answer.is_null(False)).count()}"
    )


@flow
def print_queries(db: Database):
    logger = get_run_logger()
    logger.info("Printing queries...")
    queries = get_queries(db)
    query_table = "| Text | Type | Timestamp | Document ID | Prompt Template | User | Model | UUID |\n"
    query_table += "|------|------|------------|-------------|-----------------|------|-------|------|\n"
    for query in queries:
        query_table += f"| {query.text} | {query.type} | {query.timestamp} | {query.document_id} | {query.prompt_template} | {query.user} | {query.model} | {query.uuid} |\n"
    logger.info(f"Queries:\n{query_table}")


@flow
def show_vespa_schema():
    logger = get_run_logger()
    logger.info("Getting Vespa schema...")
    logger.info(f"Vespa schema: {VespaController().get_document_schema()}")


if __name__ == "__main__":
    db = get_db()
    init_db_and_tables(db)
    show_db_stats(db)
    db.close()
