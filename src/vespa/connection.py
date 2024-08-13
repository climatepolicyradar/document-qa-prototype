from vespa.application import Vespa
from src.logger import get_logger

logger = get_logger(__name__)


def connect_to_vespa_cloud() -> Vespa:
    """
    Connect to Vespa using the configuration in the environment.

    :raises ValueError: if any environment variables are missing
    :raises ConnectionError: if the connection to Vespa fails
    :return Vespa: Vespa application object
    """
    from src.config import VESPA_URL, VESPA_CERT, VESPA_KEY

    logger.info(
        f"Connecting to Vespa at {VESPA_URL} with cert {VESPA_CERT} and key {VESPA_KEY}"
    )

    if any(var is None for var in [VESPA_URL, VESPA_CERT, VESPA_KEY]):
        raise ValueError(
            "Missing Vespa configuration in environment. Ensure VESPA_URL, VESPA_CERT_LOCATION, and VESPA_KEY_LOCATION environment variables are set and try again."
        )

    app = Vespa(url=VESPA_URL, cert=VESPA_CERT, key=VESPA_KEY)  # type: ignore

    # get_application_status() will return None if the connection fails
    if (
        app.get_application_status() is None
        or app.get_application_status().status_code != 200  # type: ignore
    ):
        raise ConnectionError(
            "Connection to Vespa failed. Please check the Vespa configuration in the environment and try again."
        )

    return app
