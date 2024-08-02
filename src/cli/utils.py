import requests


def is_api_running() -> bool:
    """Use the health check endpoint to check if the API is running."""
    try:
        r = requests.get("http://0.0.0.0:8000/health")

        return r.status_code == 200

    except requests.exceptions.ConnectionError:
        return False
