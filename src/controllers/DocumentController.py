from datetime import datetime
from cpr_data_access.models import BaseDocument, BaseMetadata
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentController:
    """Controller for handling documents."""

    def __init__(self):
        def load_metadata() -> pd.DataFrame:
            """Load document metadata from a CSV file."""
            csv_path = Path("data/docs_metadata.csv")
            assert csv_path.exists(), "📄 CSV file does not exist"
            logger.info("📄 Loading document metadata from CSV")
            return pd.read_csv(csv_path)

        self.metadata_df = load_metadata()

    def get_metadata(self, document_id: str) -> dict:
        """Get metadata for a document."""
        return self.metadata_df[
            self.metadata_df["Internal Document ID"] == document_id
        ].to_dict("records")[0]

    def create_base_document(self, document_id: str) -> BaseDocument:
        """
        Flesh out a document with the necessary information.

        Assumes the document has a document ID only.

        TODO: For the moment and for simplicity, I'm pulling this data from a CSV dump of the CPR database. But in the future we want a better way to do this. Is this data in vespa? Or grab it from something else?
        """
        # document_id column is "Internal Document ID"
        # document_name column is "Document Title"
        # document_source_url column is f"https://app.climatepolicyradar.org/document/["Document ID"]

        document_row = self.metadata_df[
            self.metadata_df["Internal Document ID"] == document_id
        ]

        assert not document_row.empty, "❌ Document ID not found in metadata"

        logger.info(f"📄 Document row: {document_row}")

        document = BaseDocument(
            document_id=document_id,
            document_name=document_row["Document Title"].values[0].strip()
            if isinstance(document_row["Document Title"].values[0], str)
            else "",
            document_source_url=f"https://app.climatepolicyradar.org/document/{document_row['Document ID'].values[0].strip()}"
            if isinstance(document_row["Document ID"].values[0], str)
            else "",
            has_valid_text=True,
            document_metadata=BaseMetadata(
                geography=document_row["Geography"].values[0].strip()
                if isinstance(document_row["Geography"].values[0], str)
                else "",
                publication_ts=datetime.now(),  # TODO PROPER DATA
            ),
            translated=True
            if document_row["Language"].values[0] != "English"
            else False,
        )

        logger.info(f"📄 Document {document.document_id} fleshed out with metadata")

        return document
