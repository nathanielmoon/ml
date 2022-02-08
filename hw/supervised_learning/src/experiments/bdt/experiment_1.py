from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'bdt/experiment_1/'


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
    plt.title('Penguin Boosted DT Accuracy by Min Samples Leaf')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'btd_accuracy_penguin_by_min_samples_leaf.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'min_samples_leaf', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='min_samples_leaf', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Boosted DT Precision by Min Samples Leaf Hyperparameter')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'btd_precision_penguin_by_min_samples_leaf.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'min_samples_leaf', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='min_samples_leaf', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('Boosted DT F1 Score by Min Samples Leaf Hyperparameter')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'btd_f1_penguin_by_min_samples_leaf.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    dt = AdaBoostClassifier(tree.DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=params['min_samples_leaf']
    ))
    dt.fit(X_train, y_train)

    def predict(X):
        return dt.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, dt


def run():
    print("Running BDT Experiment 1 ...")
    upsert_directory(OUTPUT)
    data = load_data()

    min_samples_leafs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50]

    iter_results = []
    for i, min_samples_leaf in enumerate(min_samples_leafs):
        print(
            f'\t({i + 1}/{len(min_samples_leafs)}) Running DT with min_samples_leaf = {min_samples_leaf}'
        )

        params = {"min_samples_leaf": min_samples_leaf}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'min_samples_leaf', min_samples_leaf)

        iter_results.append((min_samples_leaf, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
