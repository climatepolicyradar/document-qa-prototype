import streamlit as st
import wandb

from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from src.models.data_models import Prompt, RAGRequest
from src.streamlit.app_helpers import (
    get_image_base64,
    _find_source_indices_in_rag_response,
    _make_text_safe_for_streamlit_write,
    _format_window_as_source,
)
from src.controllers.RagController import RagController
from src.controllers.ScenarioController import ScenarioController, Scenario
from src.controllers.DocumentController import DocumentController
from src import config
from src.logger import get_logger

load_dotenv(find_dotenv())

LOGGER = get_logger(__name__)

if config.WANDB_ENABLED and not wandb.run:
    LOGGER.info("Enabling weights and biases logging...")
    wandb.init(project=config.WANDB_PROJECT_NAME)

@st.cache_resource
def get_rag_controller():
    return RagController()

@st.cache_resource
def get_document_controller():
    return DocumentController()

@st.cache_data
def get_available_documents():
    return rc.get_available_documents()

rc = get_rag_controller()
dc = get_document_controller()

image_path = "src/streamlit/img/cpr_logo_dark.png"
image_base64 = get_image_base64(image_path)

st.markdown(
    f"""
    <style>
    .container {{
        width: 100%;
        margin: 0 auto;
    }}
    .full-width-image {{
        width: 25%;
        height: auto;
    }}
    </style>
    <div class="container">
        <img src="data:image/png;base64,{image_base64}" class="full-width-image">
    </div>
    """,
    unsafe_allow_html=True,
)


st.session_state.start_run = False
st.session_state.inference_engine = "openai"

st.title("Answer questions and summarise search results from a policy document")
st.write(
    """
In this prototype you can:

- search or question a document from our database of national climate laws, policies and submissions to the UNFCCC
- summarise the answers

*This is a prototype for use in internal user testing. All queries are logged for research purposes and will not be shared externally. Enter your name to begin.*
    """
)

user = st.text_input("Your name:")

document_data = get_available_documents()

document_data_strings = [
    f"{doc.document_name} ({doc.document_metadata.geography})"
    for doc in document_data
]

document_id = None

if user:
    selected_doc = st.selectbox("Select a document to research", document_data_strings)

    if selected_doc:
        matching_doc = next(
            (
                doc
                for doc in document_data
                if f"{doc.document_name} ({doc.document_metadata.geography})" == selected_doc
            ),
            None,
        )
        assert matching_doc is not None, "🛑 No matching document found."
        document_id = matching_doc.document_id

query = st.text_input("Enter your search term or questions here") if user else None


if user:
    col1, col3 = st.columns([4, 1])
    with col1:
        st.session_state.inference_engine = st.selectbox(
            label="Inference engine",
            options=["openai", "titan", "huggingface", "tgi"],
            index=0,
        )
    with col3:
        st.write("####")  # align the run button with the dropdown
        if st.button("Run", use_container_width=True):
            st.session_state.start_run = True


if (
    user
    and query
    and document_id
    and st.session_state.start_run
    and st.session_state.inference_engine
):
    LOGGER.info(
        f"RUN PARAMETERS: User: {user} | Query: {query} | Document ID: {document_id}"
    )

    request_scenario = Scenario(
        model="gpt-4o",
        document=dc.create_base_document(document_id),
        top_k_retrieval_results=10,
        prompt=Prompt.from_template(
            "response/FAITHFULQA_SCHIMANSKI_CITATION_QA_TEMPLATE_MODIFIED"
        ),
        generation_engine=st.session_state.inference_engine,  #  type: ignore
    )

    rag_results = rc.run_rag_pipeline(query, request_scenario)

    st.markdown(_make_text_safe_for_streamlit_write(rag_results.rag_response.text))

    retrieved_text_by_start_index = {
        idx: _make_text_safe_for_streamlit_write(doc["metadata"]["text_block_window"])
        for idx, doc in enumerate(rag_results.rag_response.retrieved_documents)
    }

    cited_sources = [
        {
            "number": idx,
            "source text": retrieved_text_by_start_index[idx],
        }
        for idx in _find_source_indices_in_rag_response(rag_results.rag_response.text)
    ]
    cited_sources = sorted(cited_sources, key=lambda x: x["number"])

    st.divider()

    if cited_sources:
        st.markdown("#### Cited passages from the document:")
        for source in cited_sources:
            # TODO: add page number back in. It's no longer in the index since the move
            # to langchain
            st.markdown(
                f"""**{[source['number']]}**: {source['source text']}""",
            )

    st.session_state.start_run = False
