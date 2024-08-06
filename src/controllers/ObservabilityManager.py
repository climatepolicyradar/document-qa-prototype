
from langfuse.callback import CallbackHandler

class ObservabilityManager:
    langfuse_handler: CallbackHandler = CallbackHandler()
    
    def get_tracing_callback(self):
        return self.langfuse_handler