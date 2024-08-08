import os
import prefect
import boto3
import requests
from prefect import task
from prefect import get_run_logger
from prefect_aws import AwsCredentials
from prefect import flow, task, get_run_logger
from prefect.utilities.context import get_flow_run_id
from platform import node, platform, python_version
from prefect.server.api.server import API_VERSION
from peewee import Database, PostgresqlDatabase
import json

from prefect.runtime import flow_run

def log_essentials() -> str:
    version = prefect.__version__
    out_str = f"Network: {node()}. Instance: {platform()}. Agent is healthy ✅️"
    out_str += f"Python = {python_version()}. API: {API_VERSION}. Prefect = {version} 🚀"
    
    logger = get_run_logger()
    logger.info(out_str)
    return out_str
    

def get_secret(key: str) -> str: 
    """Returns a secret -- selecting from env if exists, otherwise query AWS SSM and add to env
    
    Queries AWS SSM for the given secret"""
    if key in os.environ:
        return os.environ[key]
    
    ssm_client = boto3.client('ssm', region_name="eu-west-1")
    
    try:
        secret = ssm_client.get_parameter(Name=f"/RAG/{key}", WithDecryption=True)
        os.environ[key] = secret["Parameter"]["Value"]
    except Exception as e:
        print(f"Failed to retrieve secret: {str(e)}")
        raise e
    
    return secret["Parameter"]["Value"]

def get_db() -> Database:
    """Retrieves the database details from AWS SSM and returns a peewee database object"""
    details = json.loads(get_secret("LABS_RDS_DB_CREDS"))
    db = PostgresqlDatabase(
        details["dbname"],
        user=details["user"],
        password=details["password"],
        host=details["host"],
        port=details["port"]
    )
    return db


@task
def send_slack_message(message):
    logger = get_run_logger()
    
    # Retrieve Slack credentials from AWS Parameter Store
    webhook_url = get_secret("PREFECT_SLACK_WEBHOOK")
    
    # Set the payload for the POST request
    payload = {
        "text": f"Flow {get_flow_run_id()}: {message}"
    }

    # Send the POST request to the Slack webhook URL
    response = requests.post(webhook_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        logger.info("Message sent successfully!")
    else:
        logger.error(f"Failed to send message. Error: {response.text}")
