# rag-labs

Labs experiment on retrieval augmented generation for in-document search

## Getting started

This project uses Python 3.9.

Installation instructions:

* run `make install`
* fill in required environment variables in `.env`

### Using the hosted database instance

To use hosted Chroma DB, you need to add its URL as a value of the `CHROMA_SERVER_HOST` environment variable.

## Running the API

```
$ uvicorn src.main:app --host 0.0.0.0 --port 8000
```

This will run the RAG app as an API, which can be reached through the `http://127.0.0.1/8000/rag/{document_id}` endpoint for generation. This is also required for the dataset creation CLI.


## Prefect flows

The repo contains two CLIs at the moment:

### Ingestion (`run_ingestion_pipeline.py`)

This runs the ingestion (aka offline) pipeline to parse documents and create the local chroma db. This is persisted to either a hosted or local chroma DB instance.

### Dataset creation (`create_generations.py`)

CLI for creating generations using the RAG API. Taking a dataframe of queries and their metadata it creates generations for each of them using a config file that determines the models and prompt templates to be used for the generations. 
The two required arguments are the input file (with the queries) and the output file that will contain the generations.

These files can be either local paths, or s3 URIs. For the latter to work, the `AWS_PROFILE` and `AWS_REGION` environment variables need to be set (as well as the aws config with access tokens).

## Running the streamlit app

The app can be run locally with

```python
make run_streamlit_demo
```

This prompts the user for a username, and then allows for querying the documents persisted in the `chroma_db_local` folder. 

There are several options for inference backend:

* `openai`: expects an `OPENAI_API_KEY` environment variable being provided
* `titan`: expects a TitanML server up and running, and port-forwarded to the local machine (by default on `127.0.0.1:3000`)
* `huggingface`: expects a `HUGGINGFACE_TOKEN` environment variable being set. Furthermore setting the `HUGGINGFACE_MODEL` variable can change the model used for serving (by default `mistralai/Mistral-7B-Instruct-v0.2`)
* `tgi`: similar to `titan` expects the TGI server running and port-forwarded to the local machine

### Setting up the inference servers on the EC2

#### Connecting

The EC2 instance is in the `eu-west-1` region on labs. You can connect to this via `ssh` - the pem file is in Bitwarden.

Example command with port forwarding and options to keep the connection alive:

``` bash
$ ssh -i ~/.ssh/titan-ec2.pem ubuntu@<ec2-url-changes-daily> -L 3000:localhost:3000 -o ServerAliveInterval=60 -o ServerAliveCountMax=3
```

After ssh'ing into the EC2 instance, we have two options: starting the Titan Takeoff or Huggingface TGI server.

#### Starting Titan Takeoff Server

Example command:
```bash
$ docker run --gpus all \
    -e TAKEOFF_CUDA_VISIBLE_DEVICES="0,1" \
    -e TAKEOFF_MODEL_NAME=TitanML/llama2-13b-chat-4bit-AWQ \
    -e TAKEOFF_DEVICE=cuda \
    -e TAKEOFF_TENSOR_PARALLEL=2 \
    -e TAKEOFF_MAX_BATCH_SIZE=1 \
    -e TAKEOFF_MAX_SEQUENCE_LENGTH=409 \
    --shm-size=2gb \
    -p 3000:3000 \
    -p 3001:3001 \
    -v ~/.iris_cache:/code/models \
    tytn/takeoff-pro:0.12.0-gpu
```

#### Starting Huggingface TGI Server

Examples command:
```bash
$ export volume=$PWD/data
$ export model=HuggingfaceH4/zephyr-7b-beta
$ docker run --gpus all \
    --shm-size 1g \
    -p 3000:80 \
    -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 \
    --model-id $model \
    --max-total-tokens 8192
```

Note, that an older version of this application is running in [labs](https://qapilot.labs.climatepolicyradar.org/).


## Datasets

Data from the `/data` folder in this repo is in labs s3 at `s3://project-rag`.

Some datasets that are related to this codebase are:

- [Machine generated dataset for Generation Policy refinement](https://huggingface.co/datasets/ClimatePolicyRadar/rag-machine-generated-dataset-mvp)


## TODOs
- [ ] Update this readme post-refactor 
- [x] Add defensive checks on vertex integration when merging for project_id and endpoint_id if they're not provided 