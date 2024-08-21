from cpr_data_access.models import BaseDocument, BaseMetadata
import logging
from src.controllers.LibraryManager import LibraryManager

logger = logging.getLogger(__name__)


class DocumentController:
    """Controller for handling documents."""

    def __init__(self):
        self.library_manager = LibraryManager()

    def get_metadata(self, document_id: str) -> dict:
        """Get metadata for a document."""
        return self.library_manager.get_document_metadata(document_id)

    def create_base_document(self, document_id: str) -> BaseDocument:
        """
        Flesh out a document with the necessary information.

        Assumes the document has a document ID only.
        """

        document_data = self.get_metadata(document_id)

        logger.info(f"📄 Document: {document_data}")

        document = BaseDocument(
            document_id=document_id,
            document_name=document_data["name"].strip(),
            document_source_url=f"https://app.climatepolicyradar.org/document/{document_data['slug'].strip()}",
            has_valid_text=True,
            document_metadata=BaseMetadata(
                geography=document_data["geography"],
                publication_ts=document_data["publication_ts"],
            ),
            translated=("English" in document_data["languages"]),
        )

        logger.info(f"📄 Document {document.document_id} fleshed out with metadata")

        return document
