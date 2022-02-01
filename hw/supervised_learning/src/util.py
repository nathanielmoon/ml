
from .paths import OUTPUT_DIR


def upsertOutputDirectory(path):
    (OUTPUT_DIR / path).mkdir(parents=True, exist_ok=True)
