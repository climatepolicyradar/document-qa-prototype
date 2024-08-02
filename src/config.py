import base64
import json
from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


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
CHROMA_DB_PATH: Path = root / "../chroma_db_local"
CHROMA_COLLECTION_NAME: str = "cpr_documents_langchain"
CHROMA_SERVER_HOST: Optional[str] = os.getenv("CHROMA_SERVER_HOST")
USE_LOCAL_CHROMA_DB: bool = _environment_variable_is_truthy("USE_LOCAL_CHROMA_DB")
PIPELINE_CACHE_PATH: Path = root / "../data/pipeline_cache"
DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", root / "../data/documents"))
WANDB_PROJECT_NAME: str = "rag-prototype-langchain"
LOGGING_LEVEL: str = os.environ.get("LOGGING_LEVEL", "DEBUG")
WANDB_ENABLED: bool = not _environment_variable_is_truthy("DISABLE_WANDB")
STREAMLIT_MOCK_GENERATION: bool = _environment_variable_is_truthy(
    "STREAMLIT_MOCK_GENERATION"
)

VESPA_URL: Optional[str] = os.getenv("VESPA_URL")
VESPA_CERT: Optional[str] = os.getenv("VESPA_CERT_LOCATION")
VESPA_KEY: Optional[str] = os.getenv("VESPA_KEY_LOCATION")

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
    "llama3-8b-chat": {"endpoint_id": "8290603546454261760", "location": "europe-west2"}
}

_assert_path_exists(DOCUMENT_DIR)
