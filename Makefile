.PHONY: install test

install:
	git init
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

install_guardrails:
	guardrails hub install hub://guardrails/detect_pii
	guardrails hub install hub://guardrails/toxic_language
	guardrails hub install hub://guardrails/web_sanitization

test:
	poetry run python -m pytest -vvv

run_streamlit_demo:
	poetry run python -m streamlit run ./src/streamlit/rag_demo_1.py

run_streamlit_analysis:
	poetry run python -m streamlit run ./src/streamlit/qa_review_app.py

build_api: export_env_vars
	docker build -t document-qa-api -f docker/Dockerfile.api .

run_api:
	docker run --env-file .env.api -p 80:80 document-qa-api 

deploy_api:
	docker tag document-qa-api:latest 845035659285.dkr.ecr.eu-west-1.amazonaws.com/document-qa-api:latest
	docker push 845035659285.dkr.ecr.eu-west-1.amazonaws.com/document-qa-api:latest

export_env_vars: 
	export $(cat .env | xargs)

prefect_login: export_env_vars
	prefect cloud login -k ${PREFECT_API_KEY}

ecr_login: export_env_vars
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}

deploy: prefect_login ecr_login
	python -m deploy