from enum import Enum


class BiEncoderModelType(Enum):
    """Models available in Vespa for retrieval using dense representations."""

    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    DISTILBERT_DOT_V5 = "sentence-transformers/msmarco-distilbert-dot-v5"
    DISTILBERT_BASE_TAS_B = "sentence-transformers/msmarco-distilbert-base-tas-b"


def get_rank_profiles() -> list[str]:
    """Get available rank profiles in Vespa."""
    biencoder_model_names = [model.name.lower() for model in BiEncoderModelType]

    dense_rank_profiles = [
        f"dense_{model_name}" for model_name in biencoder_model_names
    ]
    hybrid_rank_profiles = [
        f"hybrid_{model_name}" for model_name in biencoder_model_names
    ]
    other_rank_profiles = ["bm25", "splade"]

    return dense_rank_profiles + hybrid_rank_profiles + other_rank_profiles


def get_query_prefix(model: BiEncoderModelType) -> str:
    """Get the query prefix for a given model that's recommended for retrieval."""
    if model in {BiEncoderModelType.BGE_SMALL, BiEncoderModelType.BGE_BASE}:
        return "Represent this sentence for searching relevant passages: "

    else:
        return ""
