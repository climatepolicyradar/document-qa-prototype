import os
from typing import Any, Mapping, Optional
from enum import Enum

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_community.llms.titan_takeoff import TitanTakeoff
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, VertexAI, VertexAIModelGarden
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import google.auth
import tempfile
import json

from src.streamlit.app_helpers import _find_source_indices_in_rag_response

from dotenv import load_dotenv, find_dotenv

from src import config

load_dotenv(find_dotenv())


class LLMTypes(Enum):
    """Supported LLM types."""

    OPENAI = "openai"
    TITAN = "titan"
    HUGGINGFACE_ENDPOINT = "huggingface"
    GEMINI = "gemini"
    HUGGINGFACE_TGI = "tgi"
    VERTEX_AI = "vertexai"


class MockRagLLM(LLM):
    """
    Mock LLM for testing RAG.

    Returns a response with citations for each of the specified passage idxs in the
    form [idx].
    """

    sleep: Optional[float] = None

    def _provide_response(self, prompt: str) -> str:
        response = "hello human coder, i'm a mock LLM simulating a response.\n"
        retrieved_passage_ids = self._get_retrieved_ids_from_prompt(prompt)

        for _id in retrieved_passage_ids:
            response += f"\n - i'm citing a(nother) thing [{_id}]"

        return response

    @staticmethod
    def _get_retrieved_ids_from_prompt(prompt: str) -> list[int]:
        return _find_source_indices_in_rag_response(prompt)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return self._provide_response(prompt)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return self._provide_response(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

def get_llm(
    type: str,
    model: Optional[str] = None,
    unfiltered: bool = False,
    override_with_mock: bool = False,
) -> BaseLanguageModel:  # type: ignore
    """
    Get the LLM to use for the retrieval.

    If `override_with_mock` is set to True, a mock LLM will be returned regardless of
    `type`. This is useful for offline testing.
    """

    if override_with_mock:
        return MockRagLLM()

    try:
        _llm_type = LLMTypes(type)  # type: ignore
    except ValueError:
        raise ValueError(f"Unknown LLM type: {type}")

    if model is not None and _llm_type in {LLMTypes.TITAN, LLMTypes.HUGGINGFACE_TGI}:
        raise ValueError(
            f"Parameter model should not be specified for {_llm_type} as it's an API with a single model."
        )

    elif _llm_type == LLMTypes.OPENAI:
        # 'seed' is part of the openai api:
        # https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter
        return ChatOpenAI(
            model=model or "gpt-3.5-turbo", temperature=0, seed=42
        )

    elif _llm_type == LLMTypes.TITAN:
        return TitanTakeoff(
            base_url=os.getenv("TITAN_URL", default="http://127.0.0.1:3000"),
        )

    elif _llm_type == LLMTypes.HUGGINGFACE_ENDPOINT:
        return HuggingFaceEndpoint(
            repo_id=model or "mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=1024,
            temperature=0.01,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
        )  #  type: ignore

    elif _llm_type == LLMTypes.GEMINI:
        _model = model or "gemini-1.5-flash-latest"
        return ChatGoogleGenerativeAI(
            model=_model,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            safety_settings=_get_safety_settings(unfiltered),
            max_retries=1,
            convert_system_message_to_human=True if "1.0" in _model else False,
        )  #  type: ignore

    elif _llm_type == LLMTypes.HUGGINGFACE_TGI:
        return HuggingFaceTextGenInference(
            inference_server_url="http://localhost:3000",
            max_new_tokens=1024,
            temperature=0.01,
        )  #  type: ignore

    elif _llm_type == LLMTypes.VERTEX_AI:
        """Right this is fucking horrifying but langchain has literally no consistent way of doing auth creds across vertex integrations which means that for the Vertex Model garden ONLY we have to do this shit. Other models you can pass a creds object but noooo why would you update all the other models to ensure a consistent interface when you could force your users to do a bunch of hacky shit just to get the same functionality you couldn't be bothered to port across from other models? 
        
        TLDR: This is a hack.
        TLDR2: VertexAIModelGarden uses google.auth.default() to authenticate with GCP only which requires either their ridiculously over-engineered cross-cloud federated identity stuff OR a file on disk that stores the creds that you put the fucking PATH to in an env var. I want to use env vars to make deployment consistent from local to cloud so we're going to generate the file as an intermediary.
        
        Base64 encode json creds into the GOOGLE_APPLICATION_CREDENTIALS_BASE64 env var for this to work
        """
        creds, project_id = google.auth.load_credentials_from_dict(config.VERTEX_CREDS)
        if not creds:
            raise ValueError("Failed to authenticate with Vertex AI.")

        # We only need the creds for the LLM construction, then we can use it with wild abandon, so we'll delete the file automatically afterwards
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            temp_file.write(json.dumps(config.VERTEX_CREDS))
            temp_file.close()

            temp_file_path = temp_file.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

            model = model or "rag-meta-llama-3-1-8b-instruct"
            location = config.VERTEX_MODEL_ENDPOINTS[model]["location"] 
            
            assert project_id is not None, f"Project for model {model} not found"
            
            if config.VERTEX_MODEL_ENDPOINTS[model]["type"] == "model_garden":
                endpoint = config.VERTEX_MODEL_ENDPOINTS[model]["endpoint_id"]

                assert endpoint is not None, f"Endpoint for model {model} not found"

                llm = VertexAIModelGarden(
                    project=project_id or "",
                    endpoint_id=endpoint or "",
                    location=location,
                    allowed_model_args=["temperature", "max_tokens"],
                    safety_settings=_get_safety_settings(unfiltered),
                    max_retries=1,
                )
                
            elif config.VERTEX_MODEL_ENDPOINTS[model]["type"] == "vertex_api":
                publisher = config.VERTEX_MODEL_ENDPOINTS[model]["publisher"]
                llm = VertexAI(
                    full_model_name=f"projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}",
                    location=location,
                    project=project_id,
                    allowed_model_args=["temperature", "max_output_tokens"],
                    safety_settings=_get_safety_settings(unfiltered),
                    max_retries=1,
                )

        return llm
    
def _get_safety_settings(unfiltered: bool) -> dict:
    if unfiltered:
        return {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    else:
        return {}