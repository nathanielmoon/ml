from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR


def generate_outputs(results, data):
    sns.set()

    # Generate full data report
    report = pd.concat([x[1] for x in results], ignore_index=True)
    report.to_csv(OUTPUT / 'report.csv')

    # Generate accuracies by kernel
    plt.clf()
    reduced_report = report[report['class'] == 'accuracy'][[
        'num_iterations', 'mode', 'precision']]
    reduced_report.rename(columns={'mode': 'split'}, inplace=True)
    reduced_report = reduced_report.pivot(
        index='num_iterations', columns='split', values='precision')
    sns.lineplot(data=reduced_report)
    plt.title('Penguin Boosted DT Accuracy by Number of Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(OUTPUT / 'btd_accuracy_penguin_by_num_iterations.png')


def run_iteration(data, params):
    X_train, X_test, y_train, y_test = data
    dt = AdaBoostClassifier(tree.DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=5
    ), n_estimators=params['num_iterations'])
    dt.fit(X_train, y_train)

    def predict(X):
        return dt.predict(X)

    report = analyze_classification(predict, X_train, X_test, y_train, y_test)

    return report, dt


def run():
    print("Running BDT Experiment 1 ...")
    upsert_directory(OUTPUT)
    data = load_data()

    num_iterations_options = list(range(1, 51))

    iter_results = []
    for i, num_iterations in enumerate(num_iterations_options):
        print(
            f'\t({i + 1}/{len(num_iterations_options)}) Running DT with min_samples_leaf = {num_iterations}'
        )

        params = {"num_iterations": num_iterations}
        iteration_report, model = run_iteration(data, params)
        iteration_report.insert(0, 'num_iterations', num_iterations)

        iter_results.append((num_iterations, iteration_report, model))

    generate_outputs(iter_results, data)
    print("\tDone.")
