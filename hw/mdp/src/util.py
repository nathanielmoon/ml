from pathlib import Path
import shutil

import numpy as np


def upsert_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_cache(path):
    upsert_directory(path)
    shutil.rmtree(path)
    upsert_directory(path)


def write_cache(dir, f, data):
    with open(dir / f, "w") as fout:
        fout.write(data)


def read_cache(dir, f):
    data = None
    with open(dir / f) as fin:
        data = fin.read()
    return data


def pretty_print_policy(policy, token_map, dim_size=12):
    p = np.array([token_map[x] for x in policy])
    return np.reshape(p, (-1, dim_size))
