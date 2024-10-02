from prefect import flow, get_run_logger
from src.flows.tasks.data_tasks import create_queries, show_db_stats
import argparse
import json

from peewee import Database
from src.flows.get_db import get_db
from src.models.data_models import Query


@flow(log_prints=True)
def ingest_queries_flow(
    json_path: str,
    db: Database,
    tag: str,
):
    logger = get_run_logger()

    show_db_stats(db)

    with open(args.json_file, "r") as file:
        queries_json = json.load(file)

    queries = []

    for query_dict in queries_json:
        try:
            query = Query(
                text=query_dict["text"],
                type=query_dict["type"],
                document_id=query_dict["document_id"],
                prompt_template=query_dict["prompt_template"],
                tag=tag,
                user=query_dict["user"],
                model=query_dict["model"],
                uuid=query_dict["uuid"],
                timestamp=query_dict["timestamp"],
            )
            queries.append(query)
        except Exception as e:
            logger.error(f"Error parsing queries: {e}")

    create_queries(queries, db)
    show_db_stats(db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest queries from JSON file")
    parser.add_argument("json_file", type=str, help="Path to the queries JSON file")
    parser.add_argument("tag", type=str, help="Tag for grouping queries together")
    args = parser.parse_args()
    db = get_db()

    ingest_queries_flow(args.json_file, get_db(), args.tag)
