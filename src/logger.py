import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """
    Get a nicely formatted logger.

    :param name: name of the logger
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler(rich_tracebacks=True))
    return logger
