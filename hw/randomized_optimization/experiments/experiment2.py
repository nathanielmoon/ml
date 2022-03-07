import time

from simple_chalk import blue, magenta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score, precision_score, f1_score

import algorithms
from loaders.penguins import load_data
from util import upsert_directory, to_snake_case
from paths import OUTPUT


algos = [
    ["Random Hill Climb", algorithms.run_rhc],
    ["Simulated Annealing", algorithms.run_sa],
    ["Genetic Algorithm", algorithms.run_ga],
]

algos_shorthands = {
    "Random Hill Climb": "RHC",
    "Simulated Annealing": "SA",
    "Genetic Algorithm": "GA",
}

algos_keys = {
    "Random Hill Climb": "random_hill_climb",
    "Simulated Annealing": "simulated_annealing",
    "Genetic Algorithm": "genetic_alg",
}

SCORE_COLUMN = "Score"
EVALUATION_COLUMN = "Evaluations"


def plot_collective_curve(curves):
    plt.clf()

    p_label = "Neural Network"

    dfs = []
    for algo, curve in curves:
        df = pd.DataFrame(curve)
        df = df.rename(columns={0: SCORE_COLUMN, 1: EVALUATION_COLUMN})
        df = df.astype("int32")
        dfs.append([algo, df])

    most_evaluations = max([x[EVALUATION_COLUMN].max() for _, x in dfs])
    collective_df = pd.DataFrame(index=range(0, most_evaluations + 1))
    for algo, df in dfs:
        df = df.rename(columns={SCORE_COLUMN: algo})
        df = df.set_index([EVALUATION_COLUMN])
        collective_df = pd.concat([collective_df, df], axis=1)

    collective_df = collective_df.fillna(method="ffill")
    collective_df = collective_df.fillna(method="bfill")

    sns.lineplot(data=collective_df)
    plt.title(f"{p_label} Fitness Curves")
    plt.xlabel(EVALUATION_COLUMN)
    plt.ylabel(SCORE_COLUMN)

    plt.savefig(OUTPUT / f"{to_snake_case(p_label)}-collective.png")


def plot_test_accuracy(fitnesses):
    plt.clf()
    p_label = "Neural Network"

    fitnesses = [[algos_shorthands[l], v] for l, v in fitnesses]
    df = pd.DataFrame(fitnesses).rename(columns={0: "Algorithm", 1: "Test Accuracy"})

    ax = sns.barplot(data=df, x="Algorithm", y="Test Accuracy")
    ax.bar_label(ax.containers[0])

    plt.title(f"{p_label} Test Accuracies")
    plt.savefig(OUTPUT / f"{to_snake_case(p_label)}-accuracies.png")


def get_convergence_point(curve):
    m = max([x[0] for x in curve])
    idx = -1
    for f, i in curve:
        if f == m:
            idx = i
            break
    return idx


def get_total_evaluations(curve):
    return max([x[1] for x in curve])


def plot_evals_to_convergence(data):
    plt.clf()
    p_label = "Neural Network"

    data = [[algos_shorthands[l], v] for l, v in data]
    df = pd.DataFrame(data).rename(columns={0: "Algorithm", 1: "Evaluations"})

    ax = sns.barplot(data=df, x="Algorithm", y="Evaluations")
    ax.bar_label(ax.containers[0])

    plt.title(f"{p_label} Evaluations to Convergence")
    plt.savefig(OUTPUT / f"{to_snake_case(p_label)}-evals.png")


def plot_runtimes(data):
    plt.clf()
    p_label = "Neural Network"

    data = [[algos_shorthands[l], v] for l, v in data]
    df = pd.DataFrame(data).rename(columns={0: "Algorithm", 1: "Time (ms)"})

    ax = sns.barplot(data=df, x="Algorithm", y="Time (ms)")
    ax.bar_label(ax.containers[0])

    plt.title(f"{p_label} Time to Run")
    plt.savefig(OUTPUT / f"{to_snake_case(p_label)}-runtime.png")


def run_nn(algo_key, X_train, X_test, y_train, y_test):
    np.random.seed(69)
    nn = mlrose.NeuralNetwork(
        hidden_nodes=(75, 50),
        activation="relu",
        algorithm=algo_key,
        max_iters=1000,
        bias=True,
        is_classifier=True,
        learning_rate=0.0001,
        early_stopping=True,
        clip_max=5,
        max_attempts=100,
        random_state=42,
        curve=True,
        schedule=mlrose.ExpDecay(),
        restarts=10,
    )

    start = time.time_ns()
    nn.fit(X_train, y_train)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    def predict(X):
        return nn.predict(X)

    train_accuracy = accuracy_score(y_train, predict(X_train))
    test_accuracy = accuracy_score(y_test, predict(X_test))
    train_f1 = f1_score(y_train, predict(X_train), average="micro")
    test_f1 = f1_score(y_test, predict(X_test), average="micro")
    train_precision = precision_score(y_train, predict(X_train), average="micro")
    test_precision = precision_score(y_test, predict(X_test), average="micro")

    convergence_evaluation = get_convergence_point(nn.fitness_curve)
    total_evaluations = get_total_evaluations(nn.fitness_curve)
    time_per_evaluation = elapsed_ms / total_evaluations

    print(f"\t\tTrain Time:         {elapsed_ms}ms")
    print(f"\t\tEvals to Converge:  {convergence_evaluation}")
    print(f"\t\tTotal Evals:        {total_evaluations}")
    print(f"\t\tTime per Eval:      {time_per_evaluation}")
    print(f"\t\tTrain Accuracy:     {train_accuracy}")
    print(f"\t\tTest Accuracy:      {test_accuracy}")
    print(f"\t\tTrain F1:           {train_f1}")
    print(f"\t\tTest F1:            {test_f1}")
    print(f"\t\tTrain Precision:    {train_precision}")
    print(f"\t\tTest Precision:     {test_precision}")

    return (
        nn,  # 0
        elapsed_ms,  # 1
        convergence_evaluation,  # 2
        total_evaluations,  # 3
        time_per_evaluation,  # 4
        train_accuracy,  # 5
        test_accuracy,  # 6
        train_f1,  # 7
        test_f1,  # 8
        train_precision,  # 9
        test_precision,  # 10
    )


def run():
    sns.set()
    sns.set_palette("Set2")
    upsert_directory(OUTPUT)
    X_train, X_test, y_train, y_test = load_data(normalize=True)
    print(f"Problem: {magenta('Neural Network')}")
    results = []
    for algorithm, _ in algos:
        print(f"\tAlgorithm: {blue(algorithm)}")
        result = run_nn(algos_keys[algorithm], X_train, X_test, y_train, y_test)
        results.append(
            (
                *result,
                algorithm,
            )
        )

    plot_collective_curve([[x[11], x[0].fitness_curve] for x in results])
    plot_test_accuracy([[x[11], x[6] * 100] for x in results])
    plot_evals_to_convergence([[x[11], x[2]] for x in results])
    plot_runtimes([[x[11], x[1]] for x in results])
