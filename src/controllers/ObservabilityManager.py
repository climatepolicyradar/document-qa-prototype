import os
from langfuse.callback import CallbackHandler
from src.logger import get_logger

LOGGER = get_logger(__name__)

class ObservabilityManager:
    
    def __init__(self):
        LOGGER.info(f"Initializing ObservabilityManager for {os.environ['LANGFUSE_HOST']}")
        self.langfuse_handler: CallbackHandler = CallbackHandler()
    
    def get_tracing_callback(self):
        return self.langfuse_handler