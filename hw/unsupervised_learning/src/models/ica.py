import time

from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis
from simple_chalk import magenta

from src.util import resplit_data


def train_with_params(data, params):
    X, y, _ = data
    ica = FastICA(**params)
    start = time.time_ns()
    ica.fit(X)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    components = ica.components_
    kurt = np.abs(kurtosis(components))
    average_kurtosis = np.average(kurt)
    median_kurtosis = np.median(kurt)

    return (
        ica,
        average_kurtosis,
        median_kurtosis,
        elapsed_ms,
    )


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Independent Components Analysis"))
    np.random.seed(seed)

    n_components = data[0].shape[1]
    results = []
    for n in range(1, n_components + 1):
        model, average_kurtosis, median_kurtosis, train_time = train_with_params(
            data, {"n_components": n, "max_iter": 1000}
        )
        results.append((n, model, average_kurtosis, median_kurtosis))
        print(
            "\t\tComponents =",
            n,
            "| Average Kurtosis =",
            average_kurtosis,
            "| Median Kurtosis =",
            median_kurtosis,
            "| Train Time (ms) =",
            train_time,
        )

    return {"results": results}


def run_evaluation(data, seed=42):
    np.random.seed(seed)
