import sys
from src.flows.hello_flow import hello_flow
from src.flows.e2e_flow import e2e_flow
import dotenv
import os
from prefect.blocks.system import JSON
from prefect.deployments import DeploymentImage

dotenv.load_dotenv()

DEFAULT_JOB_VARIABLES = JSON.load("default-job-variables-prefect-mvp-labs").value
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY")

all_flows = [
    hello_flow,
    e2e_flow
]

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
    ) # type: ignore

