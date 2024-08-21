from prefect import flow, task
from art import text2art, aprint
from src.flows.utils import log_essentials, get_secret, send_slack_message
from src.logger import get_logger
import time

logger = get_logger(__name__)


@flow
def hello_flow():
    deets = log_essentials()
    send_slack_message(deets)
    send_slack_message("Hello, world!")
    send_slack_message(get_secret("test_secret"))


def sky_write(msg: str, time_unit=5):
    ascii_art = text2art("INIT")
    logger.info(f"{ascii_art}")
    time.sleep(1)
    logger.info(text2art("5"))
    time.sleep(1)
    logger.info(text2art("4"))
    time.sleep(1)
    logger.info(text2art("3"))
    time.sleep(1)
    logger.info(text2art("2"))
    time.sleep(1)
    logger.info(text2art("1"))
    time.sleep(1)
    logger.info(aprint("happy"))
    logger.info(f"{text2art(msg)}")

    ascii_art = text2art(msg)
    task_runs = []
    lines = ascii_art.split("\n")

    col_length = len(lines[0])
    print([len(line) for line in lines])

    for col in range(col_length):
        for row in lines:
            if col < len(row):
                logger.info(f"{row[col]}")
                # task_name = f"pixel_{row}_{col}"
                dummy_task.submit(
                    f"{col}, {row[col]}", time_unit, crash=(row[col] == " ")
                )
                time.sleep(0.05)

        time.sleep(time_unit + 5)

    return task_runs


@task
def dummy_task(name: str, length: int, crash: bool = False):
    print(f"Executing task: {name}")
    if crash:
        return ""
        # raise Exception("💥 CRASHED")
    time.sleep(length)


@flow
def ascii_art_flow(msg: str):
    sky_write(msg, 5)


if __name__ == "__main__":
    ascii_art_flow("beep beep")
