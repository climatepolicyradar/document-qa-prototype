import src.flows.hello_flow as hello_flow
import dotenv
import os
from prefect.blocks.system import JSON
from prefect.deployments import DeploymentImage

dotenv.load_dotenv()

DEFAULT_JOB_VARIABLES = JSON.load("default-job-variables-prefect-mvp-labs").value
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY")

hello_flow.hello_flow.deploy(
    "rag-hello-world",
    work_pool_name="mvp-labs-ecs",
    work_queue_name="mvp-labs",
    job_variables=DEFAULT_JOB_VARIABLES,
    tags=["repo:document-qa-prototype"],
    description="Hello world flow deployed from: document-qa-prototype",
    image=DeploymentImage(
        name=f"{DOCKER_REGISTRY}/prefect-rag-labs",
        tag="latest",
        dockerfile="Dockerfile"
    ),
) # type: ignore