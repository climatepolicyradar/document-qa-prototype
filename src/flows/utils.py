import os
import prefect
import boto3
import requests
from prefect import task
from prefect import get_run_logger
from prefect.utilities.context import get_flow_run_id
from platform import node, platform, python_version
from prefect.server.api.server import API_VERSION
from peewee import Database, PostgresqlDatabase
import json

from prefect_aws import AwsCredentials

aws_credentials_block = AwsCredentials.load("aws-credentials-block-labs")


def log_essentials() -> str:
    version = prefect.__version__
    out_str = f"Network: {node()}. Instance: {platform()}. Agent is healthy ✅️"
    out_str += (
        f"Python = {python_version()}. API: {API_VERSION}. Prefect = {version} 🚀"
    )

    logger = get_run_logger()
    logger.info(out_str)
    return out_str


def get_secret(key: str) -> str:
    """
    Returns a secret -- selecting from env if exists, otherwise query AWS SSM and add to env

    Queries AWS SSM for the given secret
    """

    if key in os.environ:
        return os.environ[key]

    ssm_client = aws_credentials_block.get_boto3_session().client(
        "ssm", region_name="eu-west-1"
    )

    try:
        secret = ssm_client.get_parameter(Name=f"/RAG/{key}", WithDecryption=True)
        os.environ[key] = secret["Parameter"]["Value"]
    except Exception as e:
        print(f"Failed to retrieve secret: {str(e)}")
        raise e

    return secret["Parameter"]["Value"]


# This is a temp hack while platform team is having good work/life balanece
def get_labs_session(set_as_default: bool = False) -> boto3.Session:
    labs_key = json.loads(get_secret("LABS_CREDS"))

    session = boto3.Session(
        aws_access_key_id=labs_key["key"],
        aws_secret_access_key=labs_key["secret"],
        region_name="eu-west-1",
    )

    if set_as_default:
        boto3.setup_default_session(
            aws_access_key_id=labs_key["key"],
            aws_secret_access_key=labs_key["secret"],
            region_name="eu-west-1",
        )

    return session


def get_db() -> Database:
    """Retrieves the database details from AWS SSM and returns a peewee database object"""
    details = json.loads(get_secret("LABS_RDS_DB_CREDS"))
    db = PostgresqlDatabase(
        details["dbname"],
        user=details["user"],
        password=details["password"],
        host=details["host"],
        port=details["port"],
    )
    return db


@task
def send_slack_message(message):
    logger = get_run_logger()

    # Retrieve Slack credentials from AWS Parameter Store
    webhook_url = get_secret("PREFECT_SLACK_WEBHOOK")

    flow_url = f"https://app.prefect.cloud/account/4b1558a0-3c61-4849-8b18-3e97e0516d78/workspace/1753b4f0-6221-4f6a-9233-b146518b4545/flows/flow/{get_flow_run_id()}"

    # Set the payload for the POST request
    payload = {"text": f"{message} (<{flow_url}|Flow details>)"}

    # Send the POST request to the Slack webhook URL
    response = requests.post(webhook_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        logger.info("Message sent successfully!")
    else:
        logger.error(f"Failed to send message. Error: {response.text}")
