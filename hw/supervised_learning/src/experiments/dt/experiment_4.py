from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'dt/experiment_4/'


def generate_outputs(results, data):
    print("\tGenerating outputs ...")
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by tp
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'class_count', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='class_count', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Accuracy by Class Count')
    plt.xlabel('Class Count')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_cc.png')

    # Generate precision by tp
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'class_count', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='class_count', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Precision by Class Count')
    plt.xlabel('Class Count')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_cc.png')

    # Generate f1 by tp
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'class_count', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='class_count', columns='split', values='f1-score')
    sns.lineplot(data=reduced_report)
    plt.title('F1 Score by Class Count')
    plt.xlabel('Class Count')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_cc.png')


def run_iteration(**params):
    X_train, X_test, y_train, y_test = load_data(
        n_classes=params['cc'],
        n_dimensions=3,
        n_samples=10000
    )

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
    print("Running DT Experiment 4 ...")
    upsert_directory(OUTPUT)

    class_counts = [2, 3, 4, 5]

    iter_results = []
    for i, cc in enumerate(class_counts):
        print(
            f"\t({i + 1}/{len(class_counts)}) Running DT with num classes = {cc}")

        iteration_report, model = run_iteration(cc=cc)

        iteration_report.insert(0, 'class_count', cc)
        iter_results.append((cc, iteration_report, model))

    generate_outputs(iter_results, None)
    print("\tDone.")
