from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsertDirectory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'svm/penguins/'


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    def predict(X):
        return knn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, knn


def run():
    print("Running KNN ...")
    upsertDirectory(OUTPUT)
    data = load_data()
    report, knn = run_iteration(data, {})
    print(report)
