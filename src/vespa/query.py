from enum import Enum

from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from sentence_transformers import SentenceTransformer


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


def load_sentence_transformer_model(model: BiEncoderModelType) -> SentenceTransformer:
    """Load a SentenceTransformer model by name."""
    return SentenceTransformer(model.value)


def get_all_sentence_transformer_models() -> dict[str, SentenceTransformer]:
    """Returns {model_name: SentenceTransformer} for all available models."""
    return {
        model.name: load_sentence_transformer_model(model)
        for model in BiEncoderModelType
    }


def get_query_prefix(model: BiEncoderModelType) -> str:
    """Get the query prefix for a given model that's recommended for retrieval."""
    if model in {BiEncoderModelType.BGE_SMALL, BiEncoderModelType.BGE_BASE}:
        return "Represent this sentence for searching relevant passages: "

    else:
        return ""


def make_request(
    vespa_app: Vespa,
    query: str,
    document_id: str,
    hits: int = 20,
    rank_profile: str = "hybrid_bge_small",
    hybrid_bm25_weight: float = 0.02,
) -> dict:
    """Make a request to Vespa for retrieval."""

    available_rank_profiles = get_rank_profiles()

    if rank_profile not in available_rank_profiles:
        raise ValueError(
            f"Invalid rank_profile. Must be one of {available_rank_profiles}"
        )

    if rank_profile not in {"splade", "bm25"}:
        model_name = "_".join(rank_profile.split("_")[1:]).upper().replace("-", "_")
        embedding_model = load_sentence_transformer_model(
            BiEncoderModelType[model_name]
        )

        query_prefix = get_query_prefix(BiEncoderModelType[model_name])
        query = query_prefix + query

        query_embedding = embedding_model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
        query_body = {
            "input.query(query_embedding)": query_embedding,
        }
        yql = f"select text_block_id, text_block, text_block_window from sources document_passage where (userQuery() or ({{targetHits:1000}}nearestNeighbor(text_embedding_{model_name},query_embedding))) and (document_import_id in ('{document_id}'))"

        if "hybrid" in rank_profile:
            query_body["input.query(bm25_weight)"] = hybrid_bm25_weight

    else:
        query_body = None
        yql = f"select text_block_id, text_block from sources document_passage where userQuery() and (document_import_id in ('{document_id}'))"

    with vespa_app.syncio() as session:
        response: VespaQueryResponse = session.query(
            yql=yql, hits=hits, query=query, ranking=rank_profile, body=query_body
        )

    return response.json
