# type: ignore
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.models.data_models import QAPair, DBQuery
from src.flows.utils import get_db

db = get_db()


# Helper functions
def extract_eval(evals: dict, eval_name: str):
    if evals == {}:
        return 0

    sub_dict = json.loads(evals[eval_name])
    result = sub_dict["score"]
    return result


def extract_vectara(evals_str):
    return extract_eval(evals_str, "vectara-faithfulness")


def extract_geval_faithfulness(evals_str):
    return extract_eval(evals_str, "g_eval-faithfulness")


def extract_policy_eval(evals_str):
    return extract_eval(evals_str, "g_eval-cpr-generation-policy")


# Analysis functions
def plot_model_performance(df):
    model_performance = (
        df.groupby("model")["vectara_score"].mean().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    model_performance.plot(kind="bar", ax=ax)
    ax.set_title("Average Score by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_prompt_vs_eval(df, query_prompt_column):
    df_sorted = (
        df.groupby(query_prompt_column)["vectara_score"]
        .median()
        .sort_values(ascending=True)
        .index
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.4)))
    sns.scatterplot(y=query_prompt_column, x="vectara_score", data=df, ax=ax)
    sns.scatterplot(y=query_prompt_column, x="geval_score", data=df, ax=ax)
    ax.set_title("Faithfulness Scores by Query Generation Prompt")
    ax.set_ylabel("Query Generation Prompt")
    ax.set_xlabel("Faithfulness Scores")
    plt.tight_layout()
    return fig


def analyze_no_answers(df, query_prompt_column):
    no_answer_condition = (
        (df["answer"].isnull()) | (df["answer"] == "") | (df["answer"] == "NULL")
    )
    no_answers = (
        df[no_answer_condition]
        .groupby(query_prompt_column)
        .size()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    no_answers.plot(kind="bar", ax=ax)
    ax.set_title("Number of No Answers by Query Generation Prompt")
    ax.set_xlabel("Query Generation Prompt")
    ax.set_ylabel("Number of No Answers")
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, no_answers.sum(), (no_answers / len(df) * 100).round(2)


def analyze_query_types(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="query_type", y="vectara_score", data=df, ax=ax)
    sns.boxplot(x="query_type", y="geval_score", data=df, ax=ax)
    sns.boxplot(x="query_type", y="policy_score", data=df, ax=ax)
    ax.set_title("Faithfulness Scores by Query Type")
    ax.set_xlabel("Query Type")
    ax.set_ylabel("Faithfulness Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def analyze_document_sources(df):
    source_counts = df["source_id"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    source_counts.plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Document Sources")
    ax.set_xlabel("Source ID")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def analyze_query_complexity(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df["query_length"], df["vectara_score"], label="Vectara")
    ax.scatter(df["query_length"], df["geval_score"], label="G-Eval")
    ax.scatter(df["query_length"], df["policy_score"], label="Policy")
    ax.set_title("Query Complexity vs. Faithfulness Score")
    ax.set_xlabel("Query Length (characters)")
    ax.set_ylabel("Faithfulness Score")
    ax.legend()
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("AI-Generated QA Review and Analysis App")

    # Input for pipeline_id (tag)
    tag = st.text_input("Enter the pipeline ID (tag):")

    if tag:
        # Query the database
        qa_pairs = (
            QAPair.select(QAPair, DBQuery)
            .join(DBQuery, on=(QAPair.query_id == DBQuery.id))
            .where(QAPair.pipeline_id == tag)
        )

        # Prepare data
        data = []
        for qa_pair in qa_pairs:
            print(qa_pair)
            evals = qa_pair.evals
            row = {
                "Question": qa_pair.question,
                "Answer": qa_pair.answer,
                "query_type": qa_pair.query_id.query_type,  # Changed from qa_pair.query.query_type
                "source_id": qa_pair.source_id,
                "model": qa_pair.model,
                "prompt": qa_pair.query_id.prompt,  # Changed from qa_pair.query.prompt
            }
            row["vectara_score"] = extract_vectara(evals)
            row["geval_score"] = extract_geval_faithfulness(evals)
            row["policy_score"] = extract_policy_eval(evals)
            row["query_length"] = len(qa_pair.question)
            data.append(row)

        if len(data) > 0:
            df = pd.DataFrame(data)
            print(df.head())

            # Display results
            st.subheader("QA Pairs")
            st.dataframe(
                df[
                    [
                        "Question",
                        "Answer",
                        "vectara_score",
                        "geval_score",
                        "policy_score",
                    ]
                ]
                .style.format(precision=3)
                .background_gradient(
                    subset=["vectara_score", "geval_score", "policy_score"],
                    axis=0,
                    low=0.0,
                    high=1.0,
                ),
                height=400,
                use_container_width=True,
                selection_mode="single-row",
            )
        else:
            st.write("No data found for the given pipeline ID.")
        """
        if selected_row:
            with st.sidebar:
                qa_pair = qa_pairs[selected_row.selected_rows[0]]
                end_to_end_generation = qa_pair.to_end_to_end_generation()
                st.markdown("# Context")
                st.markdown(end_to_end_generation.rag_response.retrieved_passages_as_string())
        """

        # Analysis section
        st.header("Data Analysis")

        analysis_options = [
            "Model Performance",
            "Prompt vs Evaluation",
            "No Answer Analysis",
            "Query Type Analysis",
            "Document Source Analysis",
            "Query Complexity vs. Performance",
            "Word Cloud of Queries",
        ]

        selected_analysis = st.multiselect(
            "Select analyses to perform:", analysis_options
        )

        for analysis in selected_analysis:
            if analysis == "Model Performance":
                st.subheader("Model Performance Analysis")
                fig = plot_model_performance(df)
                st.pyplot(fig)

            elif analysis == "Prompt vs Evaluation":
                st.subheader("Prompt vs Evaluation Analysis")
                fig = plot_prompt_vs_eval(df, "prompt")
                st.pyplot(fig)

            elif analysis == "No Answer Analysis":
                st.subheader("No Answer Analysis")
                fig, total_no_answers, percentage_no_answers = analyze_no_answers(
                    df, "prompt"
                )
                st.pyplot(fig)
                st.write(f"Total number of no answers: {total_no_answers}")
                st.write("Percentage of no answers for each prompt:")
                st.write(percentage_no_answers)

            elif analysis == "Query Type Analysis":
                st.subheader("Query Type Analysis")
                fig = analyze_query_types(df)
                st.pyplot(fig)

            elif analysis == "Document Source Analysis":
                st.subheader("Document Source Analysis")
                fig = analyze_document_sources(df)
                st.pyplot(fig)

            elif analysis == "Query Complexity vs. Performance":
                st.subheader("Query Complexity vs. Performance")
                fig = analyze_query_complexity(df)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
