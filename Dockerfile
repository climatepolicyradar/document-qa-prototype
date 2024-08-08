FROM prefecthq/prefect:2-python3.10
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN /usr/local/bin/python -m pip install poetry 
COPY pyproject.toml .
RUN poetry install