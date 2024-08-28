import os
from prefect import get_run_logger, task
import psutil
import boto3

from src.flows.utils import get_labs_session

# TODO PR this into CPR SDK to allow session to be passed in


def get_json_filenames(
    bucket_name: str,
    directory_path: str = "",
    session: boto3.Session = get_labs_session(),
) -> list[str]:
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    json_filenames = []

    # Ensure the directory path ends with a '/' if it's not empty
    if directory_path and not directory_path.endswith("/"):
        directory_path += "/"

    try:
        # Paginate through the objects in the bucket
        for page in paginator.paginate(Bucket=bucket_name, Prefix=directory_path):
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Check if the file has a .json extension
                    if obj["Key"].lower().endswith(".json"):
                        json_filenames.append(obj["Key"])

        return json_filenames
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


@task
def get_doc_ids_from_s3(
    bucket_name: str = "project-rag", prefix: str = "data/cpr_embeddings_output"
) -> list[str]:
    logger = get_run_logger()
    logger.info(f"🚀 Getting doc ids from s3 with prefix: {prefix}")
    logger.info(
        f"Memory before task: {psutil.Process(os.getpid()).memory_info()[0] / float(1024 * 1024)}MiB"
    )

    prefixes = get_json_filenames(bucket_name, prefix, get_labs_session())
    logger.info(f"🚀 Got {len(prefixes)} prefixes")

    doc_ids = [file.split("/")[-1].rstrip(".json") for file in prefixes]

    logger.info(f"🚀 Got {len(doc_ids)} doc ids")
    return doc_ids


def get_file_from_s3(
    bucket_name: str, file_path: str, session: boto3.Session = get_labs_session()
) -> str:
    logger = get_run_logger()
    logger.info(f"🚀 Getting file from s3: {bucket_name}/{file_path}")
    s3 = session.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_path)
    return response["Body"].read().decode("utf-8")
