import os
from langfuse.callback import CallbackHandler
from src.logger import get_logger
from src.flows.utils import get_secret

LOGGER = get_logger(__name__)


class ObservabilityManager:
    """Manager for observability tools"""

    def __init__(self):
        """Initialize the observability manager"""
        # Load env vars
        get_secret("LANGFUSE_HOST")
        get_secret("LANGFUSE_SECRET_KEY")
        get_secret("LANGFUSE_PUBLIC_KEY")

        if "LANGFUSE_HOST" not in os.environ:
            LOGGER.warning("LANGFUSE_HOST is not set")
            return

        LOGGER.info(
            f"Initializing ObservabilityManager for {os.environ['LANGFUSE_HOST']}"
        )
        self.langfuse_handler: CallbackHandler = CallbackHandler()

    def get_tracing_callback(self) -> CallbackHandler:
        """Returns the tracing callback for Langfuse"""
        return self.langfuse_handler
