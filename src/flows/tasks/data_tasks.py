import random
from prefect import task, get_run_logger, flow

from peewee import Database, fn

from src.controllers.VespaController import VespaController
from src.models.data_models import EndToEndGeneration, Notebook, Query, QAPair, DBQuery
from src.flows.get_db import get_db
from prefect.tasks import exponential_backoff


def migrate_db(db: Database):
    logger = get_run_logger()
    db.connect()
    logger.info("Creating tables...")
    db.create_tables([DBQuery, QAPair, Notebook], safe=True)
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


def get_query_by_id(db: Database, id: int) -> Query:
    logger = get_run_logger()
    query = DBQuery.get_or_none(DBQuery.id == id)
    logger.info(f"🎲 Got query with id {id}")
    return query.to_query()


def get_unanswered_queries(
    db: Database, tag: str, query_tag: str, model: str, prompt: str
) -> list[Query]:
    logger = get_run_logger()
    logger.info(
        f"Getting unanswered queries for tag {tag} with model {model} and prompt {prompt}"
    )
    query_models = (
        QAPair.select(QAPair.query_id)
        .where(QAPair.pipeline_id == tag)
        .where(QAPair.model == model)
        .where(QAPair.prompt == prompt)
        .distinct()
    )

    query_id_list = [query_model.query_id for query_model in query_models]

    logger.info(f"🎲 {query_models} ")

    queries = [
        query.to_query()
        for query in DBQuery.select().where(
            DBQuery.tag == query_tag, DBQuery.id.not_in(query_id_list)
        )
    ]

    # Randomise query order
    random.shuffle(queries)

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
def save_answer(
    tag: str, answer: EndToEndGeneration, db: Database, query: Query
) -> QAPair:
    logger = get_run_logger()
    logger.info("Saving answer to database...")
    assert query.db_id is not None, "Query database ID is not set!"
    result = answer.to_db_model(tag, query_id=str(query.db_id))
    result.save()
    return result


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


def get_answers_needing_evals(tag: str, limit: int = 10) -> list[QAPair]:
    logger = get_run_logger()
    answers = [
        qa
        for qa in QAPair.select()
        .where(QAPair.pipeline_id == tag)
        .where(QAPair.evals == {})
        .order_by(fn.Random())
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
def init_db_and_tables(db: Database):
    logger = get_run_logger()
    logger.info("Migrating database...")
    migrate_db(db)
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
