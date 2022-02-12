import time

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory, resplit_data
from src.paths import OUTPUT_DIR


def generate_outputs(report):
    sns.set()

    # train size train accuracy graph
    plt.clf()
    reduced_report = report.pivot("Train Percent", "Model", "Train Accuracy")
    print(reduced_report)
    sns.lineplot(data=reduced_report)
    plt.title('Train Accuracy by Train Percentage')
    plt.xlabel('Train Percentage')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT_DIR / 'train_accuracy_by_train_size.png')

    # train size test accuracy graph
    plt.clf()
    reduced_report = report.pivot("Train Percent", "Model", "Test Accuracy")
    print(reduced_report)
    sns.lineplot(data=reduced_report)
    plt.title('Test Accuracy by Train Percentage')
    plt.xlabel('Train Percentage')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT_DIR / 'test_accuracy_by_train_size.png')

    # train size train time graph
    plt.clf()
    reduced_report = report.pivot("Train Percent", "Model", "Train Time")
    print(reduced_report)
    sns.lineplot(data=reduced_report)
    plt.title('Train Time by Train Percentage')
    plt.xlabel('Train Percentage')
    plt.ylabel('Time (Milliseconds)')
    plt.savefig(OUTPUT_DIR / 'train_time_by_train_size.png')

    # train size query time graph
    plt.clf()
    reduced_report = report.pivot("Train Percent", "Model", "Query Time")
    print(reduced_report)
    sns.lineplot(data=reduced_report)
    plt.title('Query Time by Train Percentage')
    plt.xlabel('Train Percentage')
    plt.ylabel('Time (Milliseconds)')
    plt.savefig(OUTPUT_DIR / 'Query_time_by_train_size.png')


def create_models():
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=4
    ))
    dt = tree.DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=5
    )
    knn = KNeighborsClassifier(
        n_neighbors=8,
        weights='distance'
    )
    nn = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        learning_rate='adaptive',
        hidden_layer_sizes=(75, 50),
        random_state=42
    )
    svm = SVC(gamma='auto', kernel='poly')

    return bdt, dt, knn, nn, svm


def train_and_eval_models(models, data, train_percent):
    test_percent = 1.0 - train_percent
    X_train, X_test, y_train, y_test = resplit_data(train_percent, *data)

    results = []
    for model, name in zip(models, ['Boosted DT', 'DT', 'KNN', 'NN', 'SVM']):
        train_start = time.time_ns()
        model.fit(X_train, y_train)
        train_end = time.time_ns()
        train_time = (train_end - train_start) / 1000 / 1000  # Microseconds

        query_start = time.time_ns()
        model.predict(X_test)
        query_end = time.time_ns()
        query_time = (query_end - query_start) / 1000 / 1000

        report = analyze_classification(
            lambda X: model.predict(X), X_train, X_test, y_train, y_test)

        train_accuracy = report[report['mode'] ==
                                'train'][report['class'] == 'accuracy']['support'].values[0]
        test_accuracy = report[report['mode'] ==
                               'test'][report['class'] == 'accuracy']['support'].values[0]

        result = {
            'Train Percent': train_percent,
            'Test Percent': test_percent,
            'Model': name,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Train Time': train_time,
            'Query Time': query_time
        }
        results.append(result)

    report = pd.DataFrame(results)
    return report


def run_iteration(data, train_size):
    models = create_models()
    report = train_and_eval_models(models, data, train_size)
    return report


def run():
    print("Running space time experiment ...")
    upsert_directory(OUTPUT_DIR)

    train_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data = load_data(normalize=True)
    iter_results = []
    for i, tp in enumerate(train_percents):
        print(
            f"\t({i + 1}/{len(train_percents)}) Running DT with train % = {tp}")

        iteration_report = run_iteration(data, tp)

        iter_results.append(iteration_report)

    report = pd.concat(iter_results, ignore_index=True)
    print(report)
    generate_outputs(report)
    print("\tDone.")
