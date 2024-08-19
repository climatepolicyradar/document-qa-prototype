"""Use to encode Google service account credentials for passing via env"""
import json
import base64
import logging
from pydantic import BaseModel, FilePath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceAccountCredentials(BaseModel):
    """Data class for service account credentials"""

    type: str
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    universe_domain: str


def load_and_encode_credentials(file_path: FilePath) -> str:
    """Load the service account credentials from a JSON file and return the base64 encoded string."""
    try:
        with open(file_path, "r") as file:
            credentials = json.load(file)
            ServiceAccountCredentials(**credentials)  # Validate the structure
            encoded_credentials = base64.b64encode(
                json.dumps(credentials).encode("utf-8")
            ).decode("utf-8")
            logger.info("Credentials successfully loaded and encoded.")
            return encoded_credentials
    except Exception as e:
        logger.error(f"Failed to load and encode credentials: {e}")
        raise


# Example usage
if __name__ == "__main__":
    encoded_credentials = load_and_encode_credentials("cpr-genai-209208195b63.json")
    print(encoded_credentials)
