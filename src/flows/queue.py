"""Functions for queueing jobs for prefect flows"""

from psycopg2 import connect
from pq import PQ
from src.flows.utils import get_secret
import json

creds = json.loads(get_secret("LABS_RDS_DB_CREDS"))
conn = connect(
    dbname=creds["dbname"],
    user=creds["user"],
    password=creds["password"],
    host=creds["host"],
    port=creds["port"],
)

pq = PQ(conn)
pq.create()


def queue_job(tag: str, payload: dict):
    queue = pq[tag]
    queue.put(payload)


def get_queue(tag: str):
    return pq[tag]
