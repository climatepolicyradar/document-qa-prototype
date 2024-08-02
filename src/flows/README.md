# Introduction

This directory contains prefect flows for the RAG project. 

Use `prefect cloud login` to login to prefect cloud. 

- To run locally use `poetry run python -m flows.flow_name` 
- If using AWS, use `export AWS_PROFILE=<profile-name>;` to set the correct profile, prefixed to the command above. 

## Architecture (WIP) 
As Prefect is an orchestrated compute layer, we need to have some assets for data persistence for the flows. 

Good practice with Prefect flows is to ensure tasks are decomposed to atomic and maximally idempotent functional units as much as possible. This allows the orchestrator to effectiely manage parellelisation and retries. As such, we should aim to have a large number of small flows, rather than a few large flows. Similarly, when thinking about our data architecture, we should aim for systems that allow for heavy concurrent read and write loads. Anyway obviously this is Postgres RDS. 

Similarly, we want to manage secrets without defaulting to env vars (which have more security risks and require more manual management). Instead, we'll use AWS Secrets Manager to store our secrets. 

## Database
The labs postgres RDS instance is used as a persistent structured datastore for flows. Use the `get_db()` function to get a connection to the database -- a Peewee database object. `LABS_RDS_DB_CREDS` in the parameter store contains the credentials for the instance. 

Entities: 
- QA-pair: ID, document_id, model, prompt, pipeline_id, question, answer, eval_faithfulness, eval_policy_alignment, eval_coherence, eval_formatting, status, created_at, updated_at, generation (serialised EndToEndGeneration)

Secrets: 
- LABS_RDS_DB_CREDS -- Credentials for the labs postgres RDS instance

## TODO
- [x] Update poetry dependencies to allow prefect and httpx to effectively co-exist