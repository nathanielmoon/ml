from pathlib import Path


def upsert_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_snake_case(s):
    s = s.lower()
    return s.replace(" ", "_")
