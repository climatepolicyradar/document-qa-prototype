"""
A notebook is a collection of answers.

- When a user adds a new query, it automatically adds to the current notebook.
- When a user adds a new query to a shared notebook, it opens a new notebook.

DB Updates:
Table notebooks:
- id
- name
- created_at
- updated_at
- is_shared
- is_deleted

Addition to QAPair table:
- notebook_id nullable
"""
from typing import Optional
from src.models.data_models import Notebook, QAPair
from fastapi import HTTPException
from src.logger import get_logger

logger = get_logger(__name__)


class NotebookController:
    """Controller for interacting with notebooks."""

    def update_notebook(
        self, notebook_uuid: Optional[str], new_answer: QAPair
    ) -> Notebook:
        """Updates the notebook with a new answer."""
        if notebook_uuid is None:
            notebook = Notebook.create_new_notebook(str(new_answer.question))
        else:
            notebook = Notebook.get_by_uuid(notebook_uuid)

        if notebook is None:
            raise HTTPException(status_code=404, detail="Notebook not found")

        if notebook.is_shared:
            # Don't add to shared notebooks
            return notebook

        notebook.save()
        new_answer.notebook_id = int(notebook.id)  # type: ignore
        new_answer.save()
        return notebook

    def share_notebook(self, notebook_uuid: str) -> Notebook:
        """Shares a notebook."""
        notebook = Notebook.get_by_uuid(notebook_uuid)

        if notebook is None:
            raise HTTPException(status_code=404, detail="Notebook not found")

        notebook.is_shared = True  # type: ignore
        notebook.save()
        return notebook

    def get_notebook_with_answers(self, notebook_uuid: str) -> Notebook:
        """Gets a notebook with all of its answers."""
        nb = Notebook.get_by_uuid(notebook_uuid)

        if nb is None:
            raise HTTPException(status_code=404, detail="Notebook not found")

        try:
            logger.info(f"Getting answers for notebook {notebook_uuid} {nb.id}")
            answers = list(QAPair.select().where(QAPair.notebook_id == nb.id))
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="There are no answers in this notebook or there was an error getting the answers",
            )

        nb.answers = answers  # type: ignore
        return nb
