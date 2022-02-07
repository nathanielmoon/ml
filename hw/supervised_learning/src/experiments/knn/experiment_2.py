from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'knn/experiment_1/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'weights', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='weights', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Accuracy by Weight Function')
    plt.xlabel('Weight Function')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_weights.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'weights', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='weights', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Precision by Weight Function')
    plt.xlabel('Weight Function')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_weights.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'weights', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='weights', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('F1 Score by Weight Function')
    plt.xlabel('Weight Function')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_weights.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data

    knn = KNeighborsClassifier(
        weights=params['weights']
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
        n_classes=3,
        n_dimensions=3,
        n_samples=1000
    )

    weight_fns = ['uniform', 'distance']

    iter_results = []
    for i, weight_fn in enumerate(weight_fns):
        print(
            f'\t({i + 1}/{len(weight_fns)}) Running KNN with weight fn = {weight_fn}'
        )

        params = {"weights": weight_fn}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'weights', weight_fn)

        iter_results.append((weight_fn, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
