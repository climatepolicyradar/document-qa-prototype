"""
Generates metadata JSON for the set of documents. Assumes that the CCLW metadata CSV is in the data folder.

Queries Vespa for the set of document IDs that we have RAG data for and then queries the CCLW metadata
for the set of documents that we have RAG data for. Outputs the metadata to a JSON file in the data folder.
TODO ADD DESCRIPTION 
"""

import json
import pandas as pd
from typing import List, Dict

from src.controllers.RagController import RagController
from src.controllers.DocumentController import DocumentController
from src.logger import get_logger

logger = get_logger(__name__)

rc = RagController()
dc = DocumentController()


def get_vespa_document_ids() -> List[str]:
    """
    Query Vespa for document IDs that have RAG data.

    Returns:
        List[str]: A list of document IDs.
    """
    return [doc.document_id for doc in rc.get_available_documents()]


def generate_document_metadata(document_ids: List[str]) -> List[Dict]:
    """
    Generate metadata for documents.

    Args:
        document_ids (List[str]): List of document IDs.
        cclw_metadata (pd.DataFrame): CCLW metadata for the documents.

    Returns:
        List[Dict]: A list of dictionaries containing metadata for each document.
    """
    metadata = []

    for doc_id in document_ids:
        logger.info(f"🔍 Searching for metadata for document ID: {doc_id}")
        doc_metadata = dc.get_metadata(doc_id)

        row_data = {
            "id": doc_id,
            "document_title": doc_metadata.get("Document Title", ""),
            "geography": doc_metadata.get("Geography", ""),
            "document_type": doc_metadata.get("Document Type", ""),
            "publication_date": doc_metadata.get("Publication Date", ""),
            "document_description": doc_metadata.get("Document Description", ""),
            "source_url": doc_metadata.get("Source URL", ""),
            "language": doc_metadata.get("Language", ""),
            "sector": doc_metadata.get("Sector", ""),
            "document_status": doc_metadata.get("Document Status", ""),
            "cpr_url": f"https://app.climatepolicyradar.org/documents/{doc_metadata.get('Document ID', '')}",
        }
        # Replace NaN values with empty strings
        row_data = {k: ("" if pd.isna(v) else v) for k, v in row_data.items()}
        logger.info(f"🧹 Cleaned metadata for document ID: {doc_id}")
        print(row_data)
        metadata.append(row_data)

    return metadata


def save_metadata_to_json(
    metadata: List[Dict], output_file: str = "data/document_metadata.json"
):
    """
    Save the document metadata to a JSON file.

    Args:
        metadata (List[Dict]): The list of document metadata dictionaries.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Document metadata saved to {output_file}")


def main():
    document_ids = get_vespa_document_ids()
    metadata = generate_document_metadata(document_ids)
    save_metadata_to_json(metadata)


if __name__ == "__main__":
    main()
