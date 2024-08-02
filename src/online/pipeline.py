from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import BasePromptTemplate
from langchain_core.documents import Document as LangChainDocument
from typing import Union

from src import config
from src.logger import get_logger
from src.models.data_models import Scenario
from src.prompts.validation import text_is_too_long_for_model
from src.online.retrieval import expand_window
from operator import itemgetter


LOGGER = get_logger(__name__)


def rag_chain(
    citation_template: BasePromptTemplate,
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    scenario: Scenario
) -> RunnableSerializable:
    """
    Creates a runnable chain that takes a query and returns an answer using the RAG pipeline.

    The chain is built up as follows:
        1. (RETRIEVAL) Retrieve documents from the retriever
            input: {"query_str": str}
            output: {"documents": List[Document]}
        2. (WINDOW) NOW DONE AS PRE-PROCESSING STEP ON INGEST INTO VESPA (TODO: Confirm with Kalyan/Matyas)
        3. (FORMATTING) Create a context string from the retrieved document windows
            input: {"windows": List[List[Document]], "query_str": str}
            output: {"context_str": str}
        4. (CITATION TEMPLATE) Pass the context string to the citation template
            input: {"context_str": str}
            output: ChatPromptValue
        5. (LLM) Pass the prompt value to the language model
            input: ChatPromptValue
            output: LLMResult
        6. (OUTPUTPARSER) Parse the output of the language model
            input: LLMResult
            output: str

    The reason for creating the chain with `RunnablePassthrough.assign()` for each element
    is that this way the chain retains their results, which can be logged and checked.

    TODO: force citations
    see docs https://python.langchain.com/docs/use_cases/question_answering/citations

    Upon invokation of the chain, it returns the following object:

    {
        "query_str": str,
        "documents": List[Document],
        "context_str": str,
        "template": ChatPromptTemplate,
        "llm_output": LLMResult,
        "answer": str
    }

    Args:
        citation_template: The template to format the retrieved documents and query string with
        llm: The language model to use
        retriever: The VectorStoreRetriever for retrieval of documents

    Returns:
        RunnableParallel: The runnable chain that is to be invoked with a query
    """
    
    rag_chain_from_retriever = (
        RunnableParallel({"documents": retriever, "query_str": itemgetter("query_str"), "document_id": itemgetter("document_id")})
        | RunnablePassthrough.assign(context_str=format_docs_into_context_str)
        | RunnablePassthrough.assign(template=citation_template)
        | RunnablePassthrough.assign(llm_output=lambda x: (print(x), llm.invoke(x["template"]))[1])
        | RunnablePassthrough.assign(
            answer=lambda x: StrOutputParser().invoke(x["llm_output"])
        )
    )
    
    return rag_chain_from_retriever


def format_docs_into_context_str(
    retrieval_output: dict[str, Union[str, list[LangChainDocument]]]
) -> str:
    """
    Merges together all the documents into a single string.

    The documents are separated by newlines and prefixed with their source ID in square brackets.
    """
    
    separator = "\n\n"
    doc_str = separator
    for source_id, document in enumerate(retrieval_output["documents"]):
        doc_str += f"{separator} [{source_id}] {document.page_content}"
    
    if text_is_too_long_for_model(doc_str):
        LOGGER.warning("Document is too long for model, it will be truncated")
        
    return doc_str
