from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'nn/experiment_1/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'min_samples_leaf', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='min_samples_leaf', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Accuracy by Min Samples Leaf Hyperparameter')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_min_samples_leaf.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'min_samples_leaf', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='min_samples_leaf', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Precision by Min Samples Leaf Hyperparameter')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_min_samples_leaf.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'min_samples_leaf', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='min_samples_leaf', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('F1 Score by Min Samples Leaf Hyperparameter')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_min_samples_leaf.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data

    nn = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        learning_rate='adaptive',
        hidden_layer_sizes=(2, 50),
        random_state=42
    )

    nn.fit(X_train, y_train)

    def predict(X):
        return nn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, nn


def run():
    print("Running NN Experiment 1 ...")
    upsert_directory(OUTPUT)
    data = load_data(normalize=True)

    min_samples_leafs = [1, 2, 3, 4, 5, 10, 20, 50]

    iter_results = []
    for i, min_samples_leaf in enumerate(min_samples_leafs):
        print(
            f'\t({i + 1}/{len(min_samples_leafs)}) Running NN with min_samples_leaf = {min_samples_leaf}'
        )

        params = {"min_samples_leaf": min_samples_leaf}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'min_samples_leaf', min_samples_leaf)

        iter_results.append((min_samples_leaf, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
