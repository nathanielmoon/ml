
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data as load_spheres
from src.analyze_model import analyze_classification
from src.util import upsertDirectory, resplit_data
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'svm/experiment_5/'


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
    X_train, X_test, y_train, y_test = load_spheres(
        n_classes=2,
        n_dimensions=params['d'],
        n_samples=10000
    )

    svm = SVC(gamma='auto', kernel='rbf')

    svm.fit(X_train, y_train)

    def predict(X):
        return svm.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, svm


def run():
    print("Running SVM Experiment 5 ...")
    upsertDirectory(OUTPUT)

    dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    iter_results = []
    for i, d in enumerate(dimensions):
        print(
            f"\t({i + 1}/{len(dimensions)}) Running SVM with dimension = {d}")

        iteration_report, model = run_iteration(d=d)

        iteration_report.insert(0, 'dimensions', d)
        iter_results.append((d, iteration_report, model))

    generate_outputs(iter_results, None)
    print("\tDone.")
