FROM python:3.11-slim

RUN mkdir /app
WORKDIR /app

ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get update
RUN apt-get install --yes gcc

# Install pip and poetry
RUN pip install --no-cache --upgrade pip
RUN pip install --no-cache "poetry==1.7.0"
COPY pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

# Copy files to image
COPY data ./data
COPY chroma_db_local ./chroma_db_local
COPY src ./src
RUN poetry install --no-interaction

ENV PYTHONPATH=/app
CMD ["streamlit", "run", "src/streamlit_demo/app.py", "--server.port=8000"]
