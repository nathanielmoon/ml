from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'knn/experiment_2/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'k', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='k', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Hypersphere KNN Accuracy by K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'knn_hs_accuracy_by_k.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'k', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='k', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('KNN Precision by K')
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'knn_precision_by_k.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'k', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='k', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('KNN F1 Score by K')
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'knn_f1_by_k.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data

    knn = KNeighborsClassifier(
        n_neighbors=params['k'],
        weights='distance'
    )

    knn.fit(X_train, y_train)

    def predict(X):
        return knn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, knn


def run():
    print("Running KNN Experiment 2 ...")
    upsert_directory(OUTPUT)
    data = load_data(
        n_classes=5,
        n_dimensions=3,
        n_samples=10000
    )

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50]

    iter_results = []
    for i, k in enumerate(ks):
        print(
            f'\t({i + 1}/{len(ks)}) Running KNN with weight fn = {k}'
        )

        params = {"k": k}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'k', k)

        iter_results.append((k, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
