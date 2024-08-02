import requests
import yaml
import pandas as pd
import random
import json
import jinja2
import hashlib

from functools import reduce
from tqdm import tqdm
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Union
from dataclasses import asdict, dataclass

import src.config as config

from src.prompts.template_building import jinja_template_loader
from src.logger import get_logger
from src.models.data_models import (
    RAGRequest,
    RAGResponse,
    EndToEndGeneration,
    Query,
)

LOGGER = get_logger(__name__)

tqdm.pandas()
random.seed(42)


def get_available_document_ids_from_database() -> list[str]:
    response = requests.get(
        url="http://0.0.0.0:8000/document_ids",
        headers={
            "Content-Type": "application/json",
        },
        verify=False,
        timeout=20,
    )

    if response.status_code != 200:
        raise ValueError(f"Error getting document IDs: {response.text}")

    return response.json()["document_ids"]


def load_queries(
    queries_path: Union[str, Path], documents_per_query: int
) -> pd.DataFrame:
    """
    Load the queries csv into a dataframe, renaming columns as needed

    :param queries_path: path to the queries file containing a 'query' column and optionally a 'document_id' column
    :param documents_per_query: number of documents to run each query against, sampled with replacement. The 'document_id' column will be ignored if this is greater than 1.
    """
    if str(queries_path).endswith(".csv"):
        df = pd.read_csv(queries_path)
    elif str(queries_path).endswith(".jsonl"):
        df = pd.read_json(queries_path, lines=True)
        df.rename(columns={"text": "query"}, inplace=True)
    else:
        raise NotImplementedError("Queries file must be a CSV or JSONL file.")

    if "document_id" in df.columns and documents_per_query > 1:
        LOGGER.warning(
            "Documents per query is greater than 1, but 'document_id' column is present in the queries file."
        )
        LOGGER.warning("Ignoring 'document_id' column in the queries file.")
        df = df.drop(columns=["document_id"])

    if "document_id" not in df.columns:
        LOGGER.warning(
            "Queries file does not have a 'document_id' column. Randomly assigning document IDs."
        )
        LOGGER.info("Getting available document IDs from the database...")
        document_ids = get_available_document_ids_from_database()
        LOGGER.info(f"Got {len(document_ids)} document IDs.")

        # If documents_per_query is greater than one, we need that many rows per query
        if documents_per_query > 1:
            LOGGER.info(
                f"Sampling {documents_per_query} documents for each of the {len(df)} queries with replacement."
            )
            df = df.loc[df.index.repeat(documents_per_query)].reset_index(drop=True)

        df["document_id"] = random.choices(document_ids, k=len(df))

    if not all(c in df.columns for c in {"document_id", "query"}):
        raise ValueError("Queries file must have columns 'document_id' and 'query'")

    df.rename(
        {c: f"query_{c}" for c in df.columns if c not in {"document_id", "query"}},
        axis=1,
        inplace=True,
    )
    return df


def get_config(config_path: Union[str, Path]) -> dict:
    """Get the configuration from the config file"""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    return config


def generate_responses(
    df: pd.DataFrame, config: dict, unfiltered: bool
) -> pd.DataFrame:
    """
    Generate responses for all rows in the dataframe.

    Returns a dataframe with columns in the initial dataframe, plus a 'generation'
    column with values that are serialised EndToEndGeneration objects.

    TODO: if this generates a better data model, it'll be easier to run validations on
    the full set of responses
    """
    df["generation"] = df.progress_apply(
        lambda x: [i.model_dump() for i in generate_for_row(x, config, unfiltered)],
        axis=1,
    )
    df = df.explode("generation", ignore_index=True)
    _validate_df(df)
    return df


def _validate_df(df: pd.DataFrame):
    """Validates that the concatenation of the outputs is correct"""
    for _, row in df.iterrows():
        if "error" in row["generation"]:
            continue
        _text = row["generation"].get("text")
        assert row["text"] == _text or (pd.isna(row["text"]) and pd.isna(_text))


def generate_for_row(
    row: pd.Series, config: dict, unfiltered: bool
) -> list[EndToEndGeneration]:
    """Generate responses for a row given all the config options"""
    args_list = create_args(config)
    outputs = []
    for args in args_list:
        rag_request = RAGRequest(
            query=row["query"], document_id=row["document_id"], **asdict(args)
        )
        try:
            r = get_rag_response(
                rag_request.query,
                rag_request.document_id,
                unfiltered=unfiltered,
                **asdict(args),
            )
            output = EndToEndGeneration(
                rag_request=rag_request, rag_response=r, config=asdict(args), error=None
            )
        except Exception as e:
            LOGGER.error(f"Error generating response: {str(e)}")
            output = EndToEndGeneration(
                rag_request=rag_request,
                rag_response=None,
                config=config,
                error=str(e),
            )

        outputs.append(output)
    return outputs

def get_rag_response(
    query: str,
    document_id: str,
    generation_engine: str,
    model: str,
    prompt_template: str,
    top_k: int = 10,
    retrieval_window: int = 1,
    unfiltered: bool = False,
) -> RAGResponse:
    """Get a RAG response for the query and return response as a dict"""

    r = requests.get(
        url=f"http://0.0.0.0:8000/rag/{document_id}",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "document_id": document_id,
            "top_k": top_k,
            "generation_engine": generation_engine,
            "model": model,
            "prompt_template": prompt_template,
            "retrieval_window": retrieval_window,
            "unfiltered": unfiltered,
        },
        verify=False,
        timeout=60,
    )
    return RAGResponse.model_validate(r.json())


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
