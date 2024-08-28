from src.flows.utils import get_secret, logger


from peewee import Database, PostgresqlDatabase


import json
import os


def get_db() -> Database:
    """Retrieves the database details from AWS SSM and returns a peewee database object"""
    if "ENVIRONMENT" in os.environ and os.environ["ENVIRONMENT"] == "prod":
        logger.info("Using production database")
        creds = get_secret("QUERIED_WEB_DB_CREDS")
    else:
        logger.info("Using labs database")
        creds = get_secret("LABS_RDS_DB_CREDS")

    if not creds:
        raise ValueError("No credentials found")

    details = json.loads(creds)
    db = PostgresqlDatabase(
        details["dbname"],
        user=details["user"],
        password=details["password"],
        host=details["host"],
        port=details["port"],
    )
    return db
