from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def upsert_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def resplit_data(train_size, X_train, X_test, y_train, y_test):
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return train_test_split(X, y, train_size=train_size, random_state=42)


def inverse_transform(model, X):
    return np.dot(X, model.components_)


def pseudo_inverse_transform(model, X):
    psuedo_inverse = np.linalg.pinv(model.components_)
    return np.dot(psuedo_inverse, X.T)


def reconstruction_error(X, X_):
    return np.sum(np.power((X - X_), 2))
