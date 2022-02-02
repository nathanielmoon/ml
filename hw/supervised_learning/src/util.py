from pathlib import Path
from .paths import OUTPUT_DIR


def upsertDirectory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
