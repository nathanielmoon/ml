from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'nn/experiment_3/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'layer_size', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='layer_size', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Penguin Neural Network Accuracy by Layer Size')
    plt.xlabel('Layer Size')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'nn_peng_accuracy_by_layer_size.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    nn = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        learning_rate='adaptive',
        hidden_layer_sizes=(2, params['layer_size']),
        random_state=42
    )

    nn.fit(X_train, y_train)

    def predict(X):
        return nn.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, nn


def run():
    print("Running NN Experiment 3 ...")
    upsert_directory(OUTPUT)
    data = load_data(normalize=True)

    layer_sizes = [5, 10, 20, 40, 75, 100]

    iter_results = []
    for i, layer_size in enumerate(layer_sizes):
        print(
            f'\t({i + 1}/{len(layer_sizes)}) Running NN with n_hidden_layers = {layer_size}'
        )

        params = {"layer_size": layer_size}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'layer_size', layer_size)

        iter_results.append((layer_size, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
