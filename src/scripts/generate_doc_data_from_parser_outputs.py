"""
Generates metadata JSON for the set of documents. Download the files from S3 to data/documents/ using aws s3 sync s3://project-rag/data/cpr_embeddings_output/ . --exclude "*" --include "*.json" from data/documents dir

- Assumes that every doc is in Vespa
- Outputs the processed data to a set of JSON files in the data folder, one for each document, with annotated metadata added.
"""

import json
from pathlib import Path
from typing import List, Dict
from src.logger import get_logger

logger = get_logger(__name__)


def process_document_files() -> None:
    """
    Process JSON files in data/documents directory.

    Returns:
        List[Dict]: List of processed document data.
    """
    docs_path = Path("data/documents")
    output_path = Path("data/documents_processed")

    logger.info(f"🔍 Scanning {docs_path} for JSON files")

    for json_file in docs_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            logger.info(f"📄 Processing {json_file.name}")

            if data["html_data"] is not None:
                logger.info(":alert: HTML FILE FOUND")

            text_blocks = process_text_blocks(data["pdf_data"]["text_blocks"])

            out_json = {
                "document_id": data["document_id"],
                "document_metadata": data["document_metadata"],
                "text_blocks": text_blocks,
            }

            with open(output_path / f"{data['document_id']}.json", "w") as f:
                json.dump(out_json, f)
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse {json_file.name}. Skipping.")
        except Exception as e:
            logger.error(f"❌ Unexpected error processing {json_file.name}: {str(e)}")


def process_text_blocks(text_blocks: List[Dict]) -> List[Dict]:
    """
    Process text blocks in a document to add context data.

    Args:
        text_blocks (List[Dict]): List of text blocks.

    Returns:
        List[Dict]: List of processed text blocks with associated metadata.
    """
    current_title = ""
    current_section = ""
    processed_text_blocks = []

    for text_block in text_blocks:
        print(text_block)
        if "type" in text_block and text_block["type"].lower() == "title":
            current_title = text_block["text"][0]
        if "type" in text_block and text_block["type"].lower() == "sectionheading":
            current_section = text_block["text"][0]
        if "type" in text_block and text_block["type"].lower() == "text":
            new_text_block = {
                "text_block_id": text_block["text_block_id"],
                "title": current_title,
                "section": current_section,
                "page": text_block["page_number"],
            }
            processed_text_blocks.append(new_text_block)
    return processed_text_blocks


def main():
    process_document_files()


if __name__ == "__main__":
    main()
