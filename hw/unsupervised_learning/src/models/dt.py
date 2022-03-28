import time

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from simple_chalk import magenta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.paths import OUTPUT_DIR
from src.util import resplit_data


class FakeDTModel:
    def __init__(self, **kwargs):
        self.features = kwargs["features"] if "features" in kwargs else []
        self.originalFeatures = (
            kwargs["originalFeatures"] if "originalFeatures" in kwargs else []
        )

    def transform(self, X):
        of = list(self.originalFeatures)
        features_idxs = [of.index(f) for f in self.features]
        return X[:, features_idxs]


def train_with_params(data, params):
    X, y, originalFeatures = data
    tree = DecisionTreeClassifier(
        **{k: v for k, v in params.items() if k != "features"}
    )
    start = time.time_ns()
    tree.fit(X, y)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    feature_importances = tree.feature_importances_
    importances = list(zip(originalFeatures, feature_importances))
    importances.sort(key=lambda item: item[1], reverse=True)
    return (
        FakeDTModel(**params, originalFeatures=originalFeatures),
        importances,
        elapsed_ms,
    )


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Decision Tree"))
    np.random.seed(seed)

    n_components = data[0].shape[1]
    _, importances, train_time = train_with_params(data, {"criterion": "entropy"})
    print(
        "\tFeatures =",
        n_components,
        "| Train Time (ms) =",
        train_time,
    )

    return {
        "n_components": n_components,
        "importances": importances,
    }


def run_evaluation(data, seed=42):
    np.random.seed(seed)
