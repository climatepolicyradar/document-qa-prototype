from src.models.vespa import get_vespa_app
from vespa.io import VespaQueryResponse

from langchain_community.retrievers import VespaRetriever
import json
from typing import Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.logger import get_logger

LOGGER = get_logger(__name__)

DOCUMENT_PASSAGES_MAX = 1000  # How many passages in a full document upper bound


class VespaController:
    """Controller for Vespa operations"""

    def query(
        self,
        query: str,
        document_id: str,
        hits: int = 20,
        rank_profile: str = "hybrid_bge_small",
    ) -> dict:
        """Query Vespa for a set of passages"""
        # Connect here not on construction so connection is not held open for long
        vespa = get_vespa_app()
        LOGGER.info(f"🔍 Vespa query: {query} for document_id: {document_id}")

        query_body = None
        yql = f"select text_block_id, text_block, text_block_window, text_block_coords, text_block_page from sources document_passage where userQuery() and (document_import_id in ('{document_id}'))"

        with vespa.syncio() as session:
            response: VespaQueryResponse = session.query(
                yql=yql, hits=hits, query=query, ranking=rank_profile, body=query_body
            )

        return response.json

    def get_available_documents(self):
        """Get the available documents in the Vespa database."""

        yql = """select * from sources document_passage where true limit 0 | all(group(document_import_id) each(output(count())));"""
        vespa = get_vespa_app()

        with vespa.syncio() as session:
            response: VespaQueryResponse = session.query(
                yql=yql,
                hits=10,
            )

        res = response.json
        return res

    def get_document_schema(self):
        """Get the schema of the Vespa database."""
        vespa = get_vespa_app()

        yql = """select * from sources document_passage where true limit 1"""
        with vespa.syncio() as session:
            response: VespaQueryResponse = session.query(
                yql=yql,
                hits=DOCUMENT_PASSAGES_MAX,
            )
        return response.json

    def get_random_document(self):
        """Get a random document from the Vespa database."""
        yql = """select text_block_id, text_block, text_block_window from sources document_passage order by rand() limit 1"""

        vespa = get_vespa_app()

        with vespa.syncio() as session:
            response: VespaQueryResponse = session.query(
                yql=yql,
                hits=DOCUMENT_PASSAGES_MAX,
            )
        return response.json

    def get_document_text(self, document_id: str):
        """
        Get the text of a document from the Vespa database.

        If there are more than 1,000 text blocks we may need to reconsider the hits parameter here...
        """
        yql = f"""select text_block_id, text_block, text_block_window from sources document_passage where (document_import_id in ('{document_id}'))
        """
        vespa = get_vespa_app()

        with vespa.syncio() as session:
            response: VespaQueryResponse = session.query(
                yql=yql,
                hits=DOCUMENT_PASSAGES_MAX,
            )
        return response.json

    def retriever(self, document_id: str, top_k: int = 6) -> "VespaRetriever":
        """Returns CPR's VespaRetriever for a chain"""
        yql = f"select text_block_id, text_block, text_block_window, text_block_coords, text_block_page from sources document_passage where userQuery() and (document_import_id in ('{document_id}'))"
        # yql = f"select text_block_id, text_block, text_block_window from sources document_passage where userQuery() and (document_import_id in ('{document_id}'))"

        vespa_query_body = {"yql": yql, "hits": top_k}

        ## TODO should content field be text_block_window or text_block?
        return CPRVespaRetriever(
            controller=self,
            body=vespa_query_body,
            content_field="text_block",
            metadata_fields=[
                "text_block_id",
                "text_block_window",
                "text_block_coords",
                "text_block_page",
            ],
            app=None,
        )


class CPRVespaRetriever(VespaRetriever):
    """
    `Vespa` retriever forked from https://api.python.langchain.com/en/latest/_modules/langchain_community/retrievers/vespa_retriever.html#VespaRetriever.get_relevant_documents_with_filter

    To use our bespoke query function. Base wasn't returning docs.

    TODO: Why do we use syncio over what's in langchain vesparetriever?
    """

    controller: VespaController
    """Vespa application to query."""
    body: Dict
    """Body of the query."""
    content_field: str
    """Name of the content field."""
    metadata_fields: Sequence[str]
    """Names of the metadata fields."""

    def _query(self, body: Dict) -> List[Document]:
        vc = VespaController()
        response = vc.query(
            query=body["query"]["query_str"],
            document_id=body["query"]["document_id"],
            hits=body["hits"],
        )

        root = response["root"]
        if "errors" in root:
            raise RuntimeError(json.dumps(root["errors"]))

        docs = []
        if "children" in root:
            for child in root["children"]:
                page_content = child["fields"]["text_block_window"]
                if self.metadata_fields == "*":
                    metadata = child["fields"]
                else:
                    metadata = {
                        mf: child["fields"].get(mf) for mf in self.metadata_fields
                    }
                metadata["id"] = child["id"]
                docs.append(Document(page_content=page_content, metadata=metadata))

        LOGGER.info(f"Retrieved from vespa {len(docs)} docs")
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        body = self.body.copy()
        body["query"] = query
        return self._query(body)

    def get_relevant_documents_with_filter(
        self, query: str, *, _filter: Optional[str] = None
    ) -> List[Document]:
        """Apply a filter to the query."""
        body = self.body.copy()
        _filter = f" and {_filter}" if _filter else ""
        body["yql"] = body["yql"] + _filter
        body["query"] = query
        return self._query(body)
