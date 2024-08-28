"""Functions for queueing jobs for prefect flows"""
from src.flows.utils import get_labs_session
from src.logger import get_logger
import json

logger = get_logger(__name__)

sqs = get_labs_session().client("sqs")


def queue_job(tag: str, payload: dict):
    # Create queue if not exists
    try:
        sqs.get_queue_url(QueueName=tag)
    except sqs.exceptions.QueueDoesNotExist:
        logger.info(f"🎟️ Creating queue {tag}")
        sqs.create_queue(QueueName=tag)

    # Use the tag-specific queue URL
    queue_url = sqs.get_queue_url(QueueName=tag)["QueueUrl"]
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(payload),
    )


def get_queue_job(tag: str, job_is_json: bool = True):
    queue_url = sqs.get_queue_url(QueueName=tag)["QueueUrl"]
    response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
    print(response["Messages"][0]["ReceiptHandle"])
    if "Messages" in response:
        if job_is_json:
            result = json.loads(response["Messages"][0]["Body"])
        else:
            result = {"body": response["Messages"][0]["Body"]}
        result["receipt_handle"] = response["Messages"][0]["ReceiptHandle"]
        return result
    else:
        return None


def mark_job_done(tag: str, receipt_handle: str):
    queue_url = sqs.get_queue_url(QueueName=tag)["QueueUrl"]
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def get_queue_attributes(tag: str):
    try:
        sqs.get_queue_url(QueueName=tag)
    except sqs.exceptions.QueueDoesNotExist:
        return {
            "ApproximateNumberOfMessages": 0,
            "ApproximateNumberOfMessagesNotVisible": 0,
        }

    queue_url = sqs.get_queue_url(QueueName=tag)["QueueUrl"]

    response = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )
    return response["Attributes"]
