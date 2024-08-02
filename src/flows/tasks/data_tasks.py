from datetime import datetime
import fnmatch
from prefect import task, get_run_logger, flow

from peewee import Database, SqliteDatabase, fn

from typing import Union
from pathlib import Path
import boto3
from src.controllers.VespaController import VespaController
from src.models.data_models import EndToEndGeneration, Query, QueryType,QAPair, DBQuery
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
    tags=["calls_db"]
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
    tags=["calls_db"]
)
def get_queries(db: Database, tag: str = None) -> list[Query]:
    logger = get_run_logger()
    queries = [query.to_query() for query in DBQuery.select().where(DBQuery.tag == tag)]
    logger.info(f"🎲 Got {len(queries)} queries with tag {tag}")
    return queries

@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"]
)
def save_answer(tag: str, answer: EndToEndGeneration, db: Database):
    logger = get_run_logger()
    logger.info(f"Saving answer to database...")
    answer.to_db_model(tag).save()
    
@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=20),
    retry_jitter_factor=1,
    timeout_seconds=600,
    tags=["calls_db"]
)
def get_answers(db: Database, tag: str) -> list[QAPair]:
    logger = get_run_logger()
    answers = [qa for qa in QAPair.select().where(QAPair.pipeline_id == tag)]
    logger.info(f"🎲 Got {len(answers)} answers")
    return answers

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
    logger.info(f"Number of QAPairs with answer: {QAPair.select().where(QAPair.answer.is_null(False)).count()}")
    
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