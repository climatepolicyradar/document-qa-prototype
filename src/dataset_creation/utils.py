import random
import hashlib

from tqdm import tqdm


from src.logger import get_logger

LOGGER = get_logger(__name__)

tqdm.pandas()
random.seed(42)


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
