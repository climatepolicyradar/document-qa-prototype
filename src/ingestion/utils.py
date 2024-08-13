import json

BLOCK_SEPARATOR = "\n\n----------------\n\n"


def transform_metadata(metadata: dict) -> dict:
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = json.dumps(value)
        elif not isinstance(value, (str, int, float, bool)):
            metadata[key] = str(value)
    return metadata
