import time

from sklearn.decomposition import PCA
import numpy as np
from simple_chalk import magenta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.paths import OUTPUT_DIR
from src.util import resplit_data


def train_with_params(data, params):
    X, y, _ = data
    pca = PCA(**params)
    start = time.time_ns()
    pca.fit(X)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    variances = pca.explained_variance_ratio_
    return pca, variances, elapsed_ms


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Principal Components Analysis"))
    np.random.seed(seed)

    n_components = data[0].shape[1]
    model, variances, train_time = train_with_params(
        data, {"n_components": n_components}
    )
    print(
        "\t\tComponents =",
        n_components,
        "| Train Time (ms) =",
        train_time,
    )

    return {
        "n_components": n_components,
        "variances": variances,
        "model": model,
    }


def run_evaluation(data, seed=42):
    np.random.seed(seed)
