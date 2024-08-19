import sys
from src.flows.generate_answers_flow import queue_answer_flow, run_answers_from_queue
import dotenv
import os
from prefect.blocks.system import JSON
from prefect.deployments.runner import DeploymentImage

dotenv.load_dotenv()

DEFAULT_JOB_VARIABLES = JSON.load("default-job-variables-prefect-mvp-labs").value
DEFAULT_JOB_VARIABLES["cpu"] = 1024
DEFAULT_JOB_VARIABLES["memory"] = 4096
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY")

# Add flows here to deploy
all_flows = [queue_answer_flow, run_answers_from_queue]

base_image = DeploymentImage(
    name=f"{DOCKER_REGISTRY}/prefect-rag-labs",
    dockerfile="docker/Dockerfile.prefect",
    tag="latest",
    stream_progress_to=sys.stdout,
    buildargs={"PREFECT_API_KEY": os.getenv("PREFECT_API_KEY")},
)

flow_args = {
    "work_pool_name": "mvp-labs-ecs",
    "work_queue_name": "mvp-labs",
    "job_variables": DEFAULT_JOB_VARIABLES,
    "tags": ["repo:document-qa-prototype", "project:rag-labs"],
    "image": base_image,
}

for curr_flow in all_flows:
    print(f"Deploying flow: {curr_flow.__name__}")
    print(flow_args)
    curr_flow.deploy(
        name=f"rag-{curr_flow.__name__}",
        description=f"{curr_flow.__name__} flow deployed from: document-qa-prototype",
        **flow_args,
    )  # type: ignore
