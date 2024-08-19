"""
Generates metadata JSON for the set of documents. Assumes that the CCLW metadata CSV is in the data folder.
Queries Vespa for the set of document IDs that we have RAG data for and then queries the CCLW metadata
for the set of documents that we have RAG data for. Outputs the metadata to a JSON file in the data folder.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.vespa.connection import connect_to_vespa_cloud
from src.controllers.DocumentController import DocumentController

def get_vespa_document_ids() -> List[str]:
    """
    Query Vespa for document IDs that have RAG data.
    
    Returns:
        List[str]: A list of document IDs.
    """
    vespa = connect_to_vespa_cloud()
    query = {
        "yql": "select documentid from sources * where true limit 1000000",
        "hits": 0,
        "summary": "id"
    }
    result = vespa.query(body=query)
    return [hit['id'] for hit in result.hits]

def get_cclw_metadata(document_ids: List[str]) -> pd.DataFrame:
    """
    Load CCLW metadata for the given document IDs.
    
    Args:
        document_ids (List[str]): List of document IDs to filter the metadata.
    
    Returns:
        pd.DataFrame: Filtered metadata for the given document IDs.
    """
    csv_path = Path("data/docs_metadata.csv")
    assert csv_path.exists(), "📄 CSV file does not exist"
    
    metadata = pd.read_csv(csv_path)
    return metadata[metadata['id'].isin(document_ids)]

def generate_document_metadata(document_ids: List[str], cclw_metadata: pd.DataFrame) -> List[Dict]:
    """
    Generate metadata for documents.
    
    Args:
        document_ids (List[str]): List of document IDs.
        cclw_metadata (pd.DataFrame): CCLW metadata for the documents.
    
    Returns:
        List[Dict]: A list of dictionaries containing metadata for each document.
    """
    dc = DocumentController()
    metadata = []
    
    for doc_id in document_ids:
        doc_metadata = cclw_metadata[cclw_metadata['id'] == doc_id].to_dict('records')[0]
        doc = dc.create_base_document(doc_id)
        
        metadata.append({
            'id': doc_id,
            'title': doc_metadata.get('title', ''),
            'country': doc_metadata.get('geography', ''),
            'document_type': doc_metadata.get('document_type', ''),
            'publication_date': doc_metadata.get('publication_date', ''),
            'num_blocks': len(doc.blocks),
            'total_tokens': sum(len(block.text.split()) for block in doc.blocks)
        })
    
    return metadata

def save_metadata_to_json(metadata: List[Dict], output_file: str = 'data/document_metadata.json'):
    """
    Save the document metadata to a JSON file.
    
    Args:
        metadata (List[Dict]): The list of document metadata dictionaries.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Document metadata saved to {output_file}")

def main():
    document_ids = get_vespa_document_ids()
    cclw_metadata = get_cclw_metadata(document_ids)
    metadata = generate_document_metadata(document_ids, cclw_metadata)
    save_metadata_to_json(metadata)

if __name__ == "__main__":
    main()
