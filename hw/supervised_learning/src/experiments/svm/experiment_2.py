
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.hypersphere import load_data as load_spheres
from src.analyze_model import analyze_classification
from src.util import upsertDirectory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'svm/experiment_2/'

# Possible kernels
# {'linear', 'poly', 'rbf', 'sigmoid'}


def generate_outputs(results, data):
    print("\tGenerating outputs ...")
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'kernel', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='kernel', columns='split', values='precision')
    sns.scatterplot(data=reduced_report, s=60)
    plt.title('Accuracy by Kernel Hyperparameter')
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'accuracy_by_kernel.png')

    # Generate precision by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'kernel', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='kernel', columns='split', values='precision')
    sns.scatterplot(data=reduced_report, s=60)
    plt.title('Precision by Kernel Hyperparameter')
    plt.xlabel('Kernel')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT / 'precision_by_kernel.png')

    # Generate f1 by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'macro avg'][[
        'kernel', 'mode', 'f1-score']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='kernel', columns='split', values='f1-score')
    sns.scatterplot(data=reduced_report, s=60)
    plt.title('F1 Score by Kernel Hyperparameter')
    plt.xlabel('Kernel')
    plt.ylabel('F1 Score')
    plt.savefig(OUTPUT / 'f1_by_kernel.png')


def run_iteration(data, **params):
    X_train, X_test, y_train, y_test = data

    svm = SVC(gamma='auto', kernel=params['kernel'])

    svm.fit(X_train, y_train)

    def predict(X):
        return svm.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, svm


def run():
    print("Running SVM Experiment 2 ...")
    upsertDirectory(OUTPUT)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    data = load_spheres(
        n_classes=3,
        n_dimensions=3,
        n_samples=1000
    )
    iter_results = []
    for i, kernel in enumerate(kernels):
        print(f"\t({i + 1}/{len(kernels)}) Running SVM with kernel = {kernel}")

        iteration_report, model = run_iteration(data, kernel=kernel)

        iteration_report.insert(0, 'kernel', kernel)
        iter_results.append((kernel, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
