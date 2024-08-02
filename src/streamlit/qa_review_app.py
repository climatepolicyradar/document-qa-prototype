import pandas as pd
import streamlit as st
import json
from peewee import *
from src.models.data_models import QAPair, EndToEndGeneration
from src.flows.utils import get_db
db = get_db()

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("AI-Generated QA Review App")

    # Input for pipeline_id (tag)
    tag = st.text_input("Enter the pipeline ID (tag):")

    if tag:
        # Query the database
        qa_pairs = QAPair.select().where(QAPair.pipeline_id == tag)

        # Display results
        data = []
        for qa_pair in qa_pairs:
            evals = qa_pair.evals
            row = { "Question": qa_pair.question, "Answer": qa_pair.answer }
            for eval_name in evals.keys():
                eval_data = evals[eval_name]
                row[eval_name] = json.loads(eval_data)['score']
                
            data.append(row)

        df = pd.DataFrame(data)
        selected_row = st.dataframe(df.style \
            .format(precision=3) \
            .background_gradient(subset=[eval_name for eval_name in evals.keys()], axis=0, low=0.0, high=1.0), height=800, use_container_width=True, selection_mode="single-row",  on_select="rerun")

        if selected_row:
            with st.sidebar:
                qa_pair = qa_pairs[selected_row.selection["rows"][0]]
                end_to_end_generation = qa_pair.to_end_to_end_generation()
                st.markdown("# Context")
                st.markdown(end_to_end_generation.rag_response.retrieved_passages_as_string())
                

if __name__ == "__main__":
    main()