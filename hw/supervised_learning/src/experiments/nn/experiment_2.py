from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'nn/experiment_2/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'n_hidden_layers', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='n_hidden_layers', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Neural Network Accuracy by Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'nn_sphere_accuracy_by_n_hidden_layers.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'n_hidden_layers', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='n_hidden_layers', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Neural Network Precision by Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'nn_sphere_precision_by_n_hidden_layers.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'n_hidden_layers', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='n_hidden_layers', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('Neural Network F1 Score by Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'nn_sphere_f1_by_n_hidden_layers.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    nn = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        learning_rate='adaptive',
        hidden_layer_sizes=(params['n_hidden_layers'], 50),
        random_state=42
    )

    nn.fit(X_train, y_train)

    def predict(X):
        return nn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, nn


def run():
    print("Running NN Experiment 2 ...")
    upsert_directory(OUTPUT)
    data = load_data(
        n_classes=5,
        n_dimensions=3,
        n_samples=10000
    )

    n_hidden_layers_options = [1, 2, 3, 4, 5, 10, 20, 30, 50]

    iter_results = []
    for i, n_hidden_layers in enumerate(n_hidden_layers_options):
        print(
            f'\t({i + 1}/{len(n_hidden_layers_options)}) Running NN with n_hidden_layers = {n_hidden_layers}'
        )

        params = {"n_hidden_layers": n_hidden_layers}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'n_hidden_layers', n_hidden_layers)

        iter_results.append((n_hidden_layers, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
