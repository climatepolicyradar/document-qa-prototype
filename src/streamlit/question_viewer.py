# type: ignore

import streamlit as st
import pandas as pd

# ruff: noqa: E501
from peewee import *  # noqa: F403
from src.models.data_models import DBQuery
from src.controllers.DocumentController import DocumentController

dc = DocumentController()


def main():
    st.set_page_config(layout="wide")  # Set the app to widescreen by default
    st.title("Query Review App")

    # Sidebar
    st.sidebar.title("Filters")

    # Set S3 prefix
    # s3_prefix = "project-rag/data/cpr_embeddings_output"

    # Get unique tag values
    tags = list(DBQuery.select(DBQuery.tag).distinct().tuples())
    tags = [tag[0] for tag in tags if tag[0] is not None]
    tags.insert(0, "All")

    # Get unique model values
    models = list(DBQuery.select(DBQuery.model).distinct().tuples())
    models = [model[0] for model in models if model[0] is not None]
    models.insert(0, "All")

    # Create filter options in sidebar
    selected_tag = st.sidebar.selectbox("Select Tag", tags)
    selected_model = st.sidebar.selectbox("Select Model", models)
    prompt_filter = st.sidebar.text_input("Filter by Prompt (optional)")

    # Pagination controls
    rows_per_page = st.sidebar.number_input(
        "Rows per page", min_value=1, max_value=1000, value=50
    )
    page_number = st.sidebar.number_input("Page", min_value=1, value=1)

    # Build query
    query = DBQuery.select()
    if selected_tag != "All":
        query = query.where(DBQuery.tag == selected_tag)
    if selected_model != "All":
        query = query.where(DBQuery.model == selected_model)
    if prompt_filter:
        query = query.where(DBQuery.prompt.contains(prompt_filter))

    # Count total rows for pagination
    total_rows = query.count()
    total_pages = (total_rows - 1) // rows_per_page + 1

    st.sidebar.text(f"Total rows: {total_rows}")
    st.sidebar.text(f"Total pages: {total_pages}")

    # Apply pagination
    query = query.paginate(page_number, rows_per_page)

    # Execute query and create DataFrame
    data = []
    for q in query:
        doc = dc.create_base_document(q.document_id)
        data.append(
            {
                "ID": q.id,
                "Prompt": q.prompt,
                "Query Text": q.text,
                "Document Title": doc.document_name,
                "Tag": q.tag,
                "Model": q.model,
            }
        )

    df = pd.DataFrame(data)

    # Display table
    st.dataframe(df, height=1000)


if __name__ == "__main__":
    main()
