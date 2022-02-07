from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory, resplit_data
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'dt/experiment_3/'


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'train_percent', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='train_percent', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Accuracy by Train Percent')
    plt.xlabel('Train Percent')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_train_percent.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'train_percent', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='train_percent', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Precision by Train Percent')
    plt.xlabel('Train Percent')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_train_percent.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'train_percent', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='train_percent', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('F1 Score by Train Percent')
    plt.xlabel('Train Percent')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_train_percent.png')


def run_iteration(data, **params):
    X_train, X_test, y_train, y_test = resplit_data(params['tp'], *data)

    dt = tree.DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=5
    )
    dt.fit(X_train, y_train)

    def predict(X):
        return dt.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, dt


def run():
    print("Running DT Experiment 3 ...")
    upsert_directory(OUTPUT)

    train_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data = load_data()
    iter_results = []
    for i, tp in enumerate(train_percents):
        print(
            f"\t({i + 1}/{len(train_percents)}) Running DT with train % = {tp}")

        iteration_report, model = run_iteration(data, tp=tp)

        iteration_report.insert(0, 'train_percent', tp)
        iter_results.append((tp, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
