import time

from sklearn.manifold import Isomap
import numpy as np
from simple_chalk import magenta

from src.util import resplit_data


def train_with_params(X, params):
    ime = Isomap(**params)
    start = time.time_ns()
    ime.fit(X)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    return (
        ime,
        ime.reconstruction_error(),
        elapsed_ms,
    )


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Isomap Embedding"))
    np.random.seed(seed)

    X, y, _ = data
    n_components = data[0].shape[1]

    results = []
    for n in range(1, n_components + 1):
        model, error, train_time = train_with_params(
            X, {"n_components": n, "n_neighbors": 5, "metric": "euclidean"}
        )
        results.append((n, model, error))
        print(
            "\t\tComponents =",
            n,
            "| Error =",
            error,
            "| Train Time (ms) =",
            train_time,
        )
    return {"results": results}


def run_evaluation(data, seed=42):
    np.random.seed(seed)
