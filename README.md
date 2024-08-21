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

You can also run the API container using the `make run_api` command after running `make build_api`. `make deploy_api` deploys the container to the ECR repository, which you will need to be logged into. 

When deploying, the script will run `poetry run python -m src.scripts.generate_document_data` to generate the metadata for the documents. This requires the CCLW CSV file to be in the `data/` folder as `docs_metadata.csv`. 

### API endpoints

#### Highlights 
The /highlights endpoint takes a document id and returns the text of the citations that most strongly relates to the assertions being made. If the citation is not a good citation, it may hallucinate text. We will aim to detect this in future but for the moment can highlight the text in the citation if it appears as that text. 

```
http post localhost:8001/highlights/89cf2bc98755d034634ae9992578d370
HTTP/1.1 200 OK
content-length: 864
content-type: application/json
date: Mon, 19 Aug 2024 10:09:15 GMT
server: uvicorn

[
    {
        "answerSubstring": "This document is about amendments to the Income Tax Act of South Africa",
        "citationSubstring": " This Act to be levied in respect of the taxable income of any public benefit organisation",
        "citations": [
            "1"
        ]
    },
    {
        "answerSubstring": ". It includes amendments to various sections of the Act, including sections 6, 11, 12F, 13quat, 18, 37B, 44, and 74 of the Eighth Schedule",
        "citationSubstring": " 7. The rate of tax referred to in section 6(1) of this Act to be levied in respect of the taxable income of any public benefit organisation that has been approved by the Commissioner for the South African Revenue Service in terms of section 30(3) of the Income Tax Act, 1962, or any recreational club that has been approved by the Commissioner for the South African Revenue Service in terms of section 30A(2) of that Act is 28 per cent",
        "citations": [
            "1"
        ]
    }
]
```

#### Evaluation
`http post localhost:8001/evaluate/89cf2bc98755d034634ae9992578d370`

The /evaluate endpoint takes a document id and returns the evaluation results for that document. Each evaluation is given a success boolean. "formatting" denotes whether the response is well formatted with citations. Faithfulness is evaluated using 3 different approaches (vectara, patronus_lynx, g_eval). We want to vote on the success of each faithfulness metric -- such that 3/3 success is our highest confidence that the response is faithful and not hallucinated, 2/3 is our second highest confidence, and so on. 1/3 should be where alarm bells start ringing and being displayed to the user. 2/3 should have a note for the user to check the response. 

```HTTP/1.1 200 OK
content-length: 1177
content-type: application/json
date: Mon, 19 Aug 2024 10:05:05 GMT
server: uvicorn

[
    {
        "comments": null,
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "rule_based",
        "score": 1.0,
        "success": true,
        "type": "formatting"
    },
    {
        "comments": null,
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "g_eval",
        "score": 0.0,
        "success": false,
        "type": "cpr-generation-policy"
    },
    {
        "comments": null,
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "g_eval",
        "score": 1.0,
        "success": true,
        "type": "faithfulness"
    },
    {
        "comments": null,
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "substring_match",
        "score": 1.0,
        "success": true,
        "type": "system_response"
    },
    {
        "comments": [
            "The answer correctly states that the document is about amendments to the Income Tax Act of South Africa.",
            "It also correctly identifies the specific sections of the Act that are being amended.",
            "However, the answer does not provide any details about the nature of the amendments, which is an important aspect of understanding what the document is about."
        ],
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "patronus_lynx",
        "score": 0.0,
        "success": false,
        "type": "faithfulness"
    },
    {
        "comments": null,
        "gen_uuid": "89cf2bc98755d034634ae9992578d370",
        "name": "vectara",
        "score": 0.4542672634124756,
        "success": false,
        "type": "faithfulness"
    }
]```


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
- [ ] Write script to pull out all TODOs from the codebase and add them to this readme
- [ ] Are the config search params still being used? How and where? 

## API
### Getting a RAG answer


## Temporary notes

```
export AWS_PROFILE=labs; /usr/local/bin/aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 845035659285.dkr.ecr.eu-west-1.amazonaws.com/prefect-rag-labs
```
`prefect cloud login`