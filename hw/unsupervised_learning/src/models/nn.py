import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from simple_chalk import magenta

from src.util import resplit_data


def train_with_params(data, params):
    nn = MLPClassifier(
        solver="adam",
        alpha=1e-5,
        learning_rate="adaptive",
        hidden_layer_sizes=(2, 50),
        random_state=42,
    )
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    X, y, _ = data

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        nn,
        X,
        y,
        cv=cv,
        return_times=True,
        scoring=make_scorer(f1_score, average="micro"),
    )

    return train_sizes, train_scores, test_scores, fit_times
