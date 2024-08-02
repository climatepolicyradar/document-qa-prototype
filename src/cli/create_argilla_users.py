from pathlib import Path
import secrets
import string
import unicodedata

import typer
from pydantic import BaseModel
import argilla as rg

from src.dataset_creation.argilla import initialise_argilla
from src.logger import get_logger

LOGGER = get_logger(__name__)


class User(BaseModel):
    """User model for creating Argilla users."""

    name: str
    email: str

    @property
    def username(self):
        """
        Return the username from the user's name.

        This is the name with spaces replaced by underscores, all lowercased, and
        accents normalised.
        """

        username = self.name.replace(" ", "_").lower()
        username = (
            unicodedata.normalize("NFKD", username)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        return username


def generate_password(length: int = 20) -> str:
    """Generate a random password."""
    alphabet = string.ascii_letters + string.digits
    password = "".join(
        secrets.choice(alphabet) for i in range(length)
    )  # for a 20-character password

    return password


def create_argilla_users(
    user_csv_path: Path,
    role: str = "annotator",
):
    """
    Create argilla users from a CSV file.

    The CSV file should have no header and the first two columns should be username and
    email. A password will be generated for each user and saved to a new CSV file.
    """

    initialise_argilla()

    if not user_csv_path.exists():
        raise FileNotFoundError(f"User CSV not found: {user_csv_path}")

    users = []

    with open(user_csv_path, "r") as f:
        for line in f.readlines():
            name, email = line.strip().split(",")
            users.append(User(name=name, email=email))

    breakpoint()
    passwords = [generate_password() for _ in range(len(users))]

    output_path_with_passwords = user_csv_path.parent / (
        user_csv_path.stem + "_with_passwords" + user_csv_path.suffix
    )

    typer.confirm(
        f"About to create {len(users)} users with usernames: {', '.join([u.username for u in users])}. Do you want to continue?",
        abort=True,
    )

    Path(output_path_with_passwords).write_text(
        "\n".join(
            [
                f"{user.username},{user.email},{password}"
                for user, password in zip(users, passwords)
            ]
        )
    )

    for user, password in zip(users, passwords):
        try:
            rg.User.create(
                username=user.username,
                first_name=" ".join(user.name.split(" ")[:-1]),
                last_name=user.name.split(" ")[-1],
                password=password,
                role=role,
            )

            LOGGER.info(f"Created user {user.username}")

        except Exception as e:
            LOGGER.error(f"Error creating user {user.username}: {e}")


if __name__ == "__main__":
    typer.run(create_argilla_users)
