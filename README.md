# rag-labs

Labs experiment on retrieval augmented generation for in-document search

## Getting started

This project uses Python 3.9.

Installation instructions:

* run `make install`
* fill in required environment variables in `.env`

## Running the API

```
$ uvicorn src.main:app --host 0.0.0.0 --port 8000
```

This will run the RAG app as an API, which can be reached through the `http://127.0.0.1/8000/rag/{document_id}` endpoint for generation. This is also required for the dataset creation CLI.

Use `--reload` for hot reloading. 

You will need to be logged in to AWS SSO so that the backend can retrieve the RDS credentials from the parameter store. Apply a logged in profile using:

```
$ aws sso login --profile <profile-name>
$ export AWS_PROFILE=<profile-name>; uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Prefect flows

TODO

## Running the streamlit apps

`src/streamlit` contains streamlit apps.

TODO


## Inference 

There are several options for inference backend:

* `openai`: expects an `OPENAI_API_KEY` environment variable being provided
* `titan`: expects a TitanML server up and running, and port-forwarded to the local machine (by default on `127.0.0.1:3000`)
* `huggingface`: expects a `HUGGINGFACE_TOKEN` environment variable being set. Furthermore setting the `HUGGINGFACE_MODEL` variable can change the model used for serving (by default `mistralai/Mistral-7B-Instruct-v0.2`)
* `tgi`: similar to `titan` expects the TGI server running and port-forwarded to the local machine
* `vertexai`: requires credentials; TODO 


## Datasets

You will need the CCLW CSV dataset in the `data/` folder as `docs_metadata.csv`. 

Data from the `/data` folder in this repo is in labs s3 at `s3://project-rag`.

Some datasets that are related to this codebase are:

- [Machine generated dataset for Generation Policy refinement](https://huggingface.co/datasets/ClimatePolicyRadar/rag-machine-generated-dataset-mvp)


## TODOs
- [ ] Update this readme post-refactor 
- [x] Add defensive checks on vertex integration when merging for project_id and endpoint_id if they're not provided 

## Temporary notes

```
export AWS_PROFILE=labs; /usr/local/bin/aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 845035659285.dkr.ecr.eu-west-1.amazonaws.com/prefect-rag-labs
```
`prefect cloud login`