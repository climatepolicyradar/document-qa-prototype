.PHONY: install test

install:
	git init
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

test:
	poetry run python -m pytest -vvv

run_ingestion_pipeline:
	poetry run python -m src.cli.run_ingestion_pipeline

run_streamlit_demo:
	poetry run python -m streamlit run ./src/streamlit_demo/app.py
