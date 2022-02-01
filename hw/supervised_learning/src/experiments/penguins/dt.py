from sklearn import tree
import matplotlib.pyplot as plt

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification


def run():
    X_train, X_test, y_train, y_test = load_data()
    dt = tree.DecisionTreeClassifier(
        random_state=42,
    )
    dt.fit(X_train, y_train)

    def predict(X):
        return dt.predict(X)

    analyze_classification(predict, X_train, X_test, y_train, y_test)

    # tree.plot_tree(dt)
    # plt.show()
