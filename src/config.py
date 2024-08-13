import base64
import json
import os
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from src.flows.utils import get_secret
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)
logger.info("Beginning configuration setup")

load_dotenv(find_dotenv(), override=True)

## Load our environment variables -- we may be running in a serverless environment, so we need to load them from AWS SSM
get_secret("VERTEX_AI_PROJECT")
get_secret("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
get_secret("GOOGLE_API_KEY")


VESPA_URL: Optional[str] = get_secret("VESPA_URL")
VESPA_CERT: Optional[str] = get_secret("VESPA_CERT_LOCATION")
VESPA_KEY: Optional[str] = get_secret("VESPA_KEY_LOCATION")

# Lol i can't believe I have to do this again why don't anyone just let you pass actual data rather than file paths
if not VESPA_CERT or VESPA_CERT == "" or not VESPA_KEY or VESPA_KEY == "":
    logger.info("VESPA_CERT or VESPA_KEY not found, downloading from AWS SSM")

    cert_content = get_secret("VESPA_CERT")
    key_content = get_secret("VESPA_KEY")

    cert_path = Path("cert.pem")
    key_path = Path("key.pem")

    with open(cert_path, "w") as cert_file:
        cert_file.write(cert_content)

    with open(key_path, "w") as key_file:
        key_file.write(key_content)

    # Assume running in docker container, which is /app workdir
    VESPA_CERT = f"/app/{cert_path}"
    VESPA_KEY = f"/app/{key_path}"

    logger.info(f"VESPA_CERT: {VESPA_CERT}, VESPA_KEY: {VESPA_KEY}")


def _assert_path_exists(path: Path):
    """Check a path exists. This should be used for all paths that can be specified from the .env file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Path specified in config or .env does not exist: {path}"
        )


def _environment_variable_is_truthy(env_var: str) -> bool:
    return os.getenv(env_var, "false").lower() in ("true", "1")


root = Path(__file__).parent

EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
WANDB_PROJECT_NAME: str = "rag-prototype-langchain"
LOGGING_LEVEL: str = os.environ.get("LOGGING_LEVEL", "DEBUG")
WANDB_ENABLED: bool = not _environment_variable_is_truthy("DISABLE_WANDB")

root_templates_folder = Path("src/prompts/prompt_templates")
response_templates_folder = root_templates_folder / "response"
policy_templates_folder = root_templates_folder / "policy"
query_templates_folder = root_templates_folder / "query"
evaluation_templates_folder = root_templates_folder / "evaluation"

VERTEX_CREDS: Optional[str] = None
encoded_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
if encoded_creds:
    VERTEX_CREDS = json.loads(base64.b64decode(encoded_creds).decode("utf-8"))

VERTEX_MODEL_ENDPOINTS = {
    "llama3-8b-chat": {
        "type": "model_garden",
        "endpoint_id": "8290603546454261760",
        "location": "europe-west2",
        "params": {"max_output_tokens": 2048},
    },
    "neural-chat-7b": {
        "type": "model_garden",
        "endpoint_id": "1766295061277966336",
        "location": "europe-west2",
        "params": {"max_tokens": 2048},
    },
    "llama3-1-8b-instruct": {
        "type": "model_garden",
        "endpoint_id": "7530902584312201216",
        "location": "europe-west2",
        "params": {"max_tokens": 2048},
    },
    "climate-gpt-7b": {  # This isn't working until we get llama2chat wrapper working.
        "type": "model_garden",
        "endpoint_id": "3318207345372168192",
        "location": "europe-west2",
        "params": {"max_tokens": 2048},
    },
    "mistral-nemo": {
        "type": "vertex_api",
        "model_name": "mistral-nemo@2407",
        "publisher": "mistralai",
        "location": "europe-west4",
        "params": {"max_output_tokens": 2048},
    },
    "patronus-lynx": {
        "type": "model_garden",
        "endpoint_id": "5923117517340934144",
        "location": "europe-west2",
        "params": {"max_tokens": 2048},
    },
}
