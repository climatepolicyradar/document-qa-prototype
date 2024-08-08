.PHONY: install test

install:
	git init
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

test:
	poetry run python -m pytest -vvv

run_streamlit_demo:
	poetry run python -m streamlit run ./src/streamlit/rag_demo_1.py

run_streamlit_analysis:
	poetry run python -m streamlit run ./src/streamlit/qa_review_app.py


export_env_vars: 
	export $(cat .env | xargs)

prefect_login: export_env_vars
	prefect cloud login -k ${PREFECT_API_KEY}

ecr_login: export_env_vars
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}

deploy: prefect_login ecr_login
	python -m deploy