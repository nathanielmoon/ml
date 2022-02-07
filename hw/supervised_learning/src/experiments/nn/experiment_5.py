from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR


OUTPUT = OUTPUT_DIR / 'nn/experiment_5/'


def generate_outputs(results, data):
    print("\tGenerating outputs ...")
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by tp
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'dimensions', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='dimensions', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Accuracy by Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_dimensions.png')

    # Generate precision by tp
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'dimensions', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='dimensions', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Precision by Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_dimensions.png')

    # Generate f1 by tp
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'dimensions', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='dimensions', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('F1 Score by Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_dimensions.png')


def run_iteration(**params):
    X_train, X_test, y_train, y_test = load_data(
        n_classes=2,
        n_dimensions=params['d'],
        n_samples=10000
    )

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
    print("Running NN Experiment 5 ...")
    upsert_directory(OUTPUT)

    dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    iter_results = []
    for i, d in enumerate(dimensions):
        print(
            f"\t({i + 1}/{len(dimensions)}) Running NN with dimension = {d}")

        iteration_report, model = run_iteration(d=d)

        iteration_report.insert(0, 'dimensions', d)
        iter_results.append((d, iteration_report, model))

    generate_outputs(iter_results, None)
    print("\tDone.")
