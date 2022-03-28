import time

from sklearn.random_projection import GaussianRandomProjection
import numpy as np
from simple_chalk import magenta

from src.util import resplit_data, inverse_transform, reconstruction_error

n = 100


def train_with_params(data, params):
    errors = []
    for i in range(n):
        X, y, _ = data
        rca = GaussianRandomProjection(**params)
        start = time.time_ns()
        rca.fit(X)
        end = time.time_ns()
        elapsed_ms = (end - start) / 1000000

        X_projected = rca.transform(X)
        X_reconstructed = inverse_transform(rca, X_projected)

        error = reconstruction_error(X, X_reconstructed)
        errors.append(error)

    return (
        rca,
        np.mean(np.array(errors)),
        errors,
        elapsed_ms,
    )


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Randomized Projection"))
    np.random.seed(seed)

    n_components = data[0].shape[1]

    results = []
    for n in range(1, n_components + 1):
        model, error, errors, train_time = train_with_params(data, {"n_components": n})
        results.append((n, model, error, errors))
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
