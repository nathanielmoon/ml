from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'nn/penguins/'


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    nn = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(10, 10),
        random_state=1
    )

    nn.fit(X_train, y_train)

    def predict(X):
        return nn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, nn


def run():
    upsert_directory(OUTPUT)
    data = load_data()
    report, nn = run_iteration(data, {})
    print(report)
