import time
import json

from simple_chalk import blue, magenta, white
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import algorithms
import problems
from util import upsert_directory, to_snake_case
from paths import OUTPUT

algos = [
    ["Random Hill Climb", algorithms.run_rhc],
    ["Simulated Annealing", algorithms.run_sa],
    ["Genetic Algorithm", algorithms.run_ga],
    ["MIMIC", algorithms.run_mimic],
]

probs = [
    ["Four Peaks", problems.create_peaks_problem, 42],
    ["Flip Flop", problems.create_flipflop_problem, 10],
    ["Continous Peaks", problems.create_continuous_peaks_problem, 42],
]

algos_shorthands = {
    "Random Hill Climb": "RHC",
    "Simulated Annealing": "SA",
    "Genetic Algorithm": "GA",
    "MIMIC": "MIMIC",
}

SCORE_COLUMN = "Score"
EVALUATION_COLUMN = "Evaluations"


def plot_curve(problem_name, algo_name, curve, length, out):
    plt.clf()

    df = pd.DataFrame(curve)
    df = df.rename(columns={0: SCORE_COLUMN, 1: EVALUATION_COLUMN})
    df = df.astype("int32")
    df.to_csv(out / f"{to_snake_case(problem_name)}-{to_snake_case(algo_name)}.csv")
    sns.lineplot(data=df, x=EVALUATION_COLUMN, y=SCORE_COLUMN)
    plt.title(f"{problem_name}: {algo_name} Learning Curve")
    # plt.savefig(
    #    out / f"{to_snake_case(problem_name)}-{to_snake_case(algo_name)}-{length}.png"
    # )

    return df


def plot_collective_curve(dfs, p_label, length, out):
    plt.clf()

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
    if p_label == "Flip Flop":
        plt.legend(loc="lower center")

    plt.savefig(out / f"{to_snake_case(p_label)}-collective-{length}.png")


def plot_times(times, length, out):
    times = [[p, algos_shorthands[a], t] for p, a, t in times]
    plt.clf()

    df = pd.DataFrame(times).rename(
        columns={0: "Problem", 1: "Algorithm", 2: "Time (ms)"}
    )

    g = sns.catplot(data=df, x="Algorithm", y="Time (ms)", hue="Problem", kind="bar")
    g.fig.set_size_inches(15, 8)
    g.fig.subplots_adjust(top=0.81, right=0.86)

    # extract the matplotlib axes_subplot objects from the FacetGrid
    ax = g.facet_axis(0, 0)

    # iterate through the axes containers
    for c in ax.containers:
        labels = [f"{v.get_height()}" for v in c]
        ax.bar_label(c, labels=labels, label_type="edge")

    plt.title(f"Total Run Times")
    plt.savefig(out / f"runtimes-{length}.png")


def plot_fitnesses(fitnesses, p_label, length, out):
    plt.clf()

    fitnesses = [[algos_shorthands[l], v] for l, v in fitnesses]
    df = pd.DataFrame(fitnesses).rename(columns={0: "Algorithm", 1: "Fitness"})

    ax = sns.barplot(data=df, x="Algorithm", y="Fitness")
    ax.bar_label(ax.containers[0])

    plt.title(f"{p_label} Optimal Fitnesses")
    plt.savefig(out / f"{to_snake_case(p_label)}-fitnesses-{length}.png")


def plot_evaluations_to_convergence(data, p_label, length, out):
    plt.clf()

    data = [[algos_shorthands[l], v] for l, v in data]
    df = pd.DataFrame(data).rename(columns={0: "Algorithm", 1: "Evaluations"})

    ax = sns.barplot(data=df, x="Algorithm", y="Evaluations")
    ax.bar_label(ax.containers[0])

    plt.title(f"{p_label} Evaluations to Convergence")
    plt.savefig(out / f"{to_snake_case(p_label)}-evals_to_convergence-{length}.png")


def get_convergence_point(curve):
    m = max([x[0] for x in curve])
    idx = -1
    for f, i in curve:
        if f == m:
            idx = i
            break
    return idx


def plot_fitness_by_space(df, out):
    for problem, _, _ in probs:
        plt.clf()
        subdf = df[df["Problem"] == problem]
        subdf = subdf[["Length", "Fitness", "Algorithm"]]
        subdf = subdf.pivot("Length", "Algorithm", "Fitness")
        print(subdf)
        sns.lineplot(data=subdf)
        plt.title(f"{problem}: Fitness by Space Size")
        plt.xlabel("Length of Bit String")
        plt.ylabel("Fitness")
        plt.savefig(OUTPUT / f"fitness-by-space_{problem}.png")


def plot_time_by_space(df, out):
    for problem, _, _ in probs:
        plt.clf()
        subdf = df[df["Problem"] == problem]
        subdf = subdf[["Length", "Time (ms)", "Algorithm"]]
        subdf = subdf.pivot("Length", "Algorithm", "Time (ms)")
        print(subdf)
        sns.lineplot(data=subdf)
        plt.title(f"{problem}: Time by Space Size")
        plt.xlabel("Length of Bit String")
        plt.ylabel("Time (ms)")
        plt.savefig(OUTPUT / f"time-by-space_{problem}.png")


def run_with_length(length, out):
    times = []
    fitness_totals = []
    print(f"Running with length: {white(str(length))}")
    for p_label, create_problem, seed in probs:
        print(f"\tProblem: {magenta(p_label)}")
        np.random.seed(seed)
        curve_dfs = []
        fitnesses = []
        evaluations_to_converge = []

        for a_label, algorithm in algos:
            print(f"\t\tAlgorithm: {blue(a_label)}")
            p, i = create_problem(length)
            p.set_mimic_fast_mode(True)

            start = time.time_ns()
            _, best_fitness, curve = algorithm(p, i, max_attempts=10, max_iters=1000000)
            print("BNEST FINENTS", best_fitness)
            end = time.time_ns()
            elapsed_ms = (end - start) / 1000000
            times.append([p_label, a_label, elapsed_ms])

            fitnesses.append([a_label, best_fitness])
            fitness_totals.append([p_label, a_label, best_fitness])
            evaluations_to_converge.append([a_label, get_convergence_point(curve)])

            df = plot_curve(p_label, a_label, curve, length, out)
            curve_dfs.append([a_label, df])

            print("\t\t\tBest Fitness:\t", best_fitness)
            print(f"\t\t\tTime Elapsed:\t\t {elapsed_ms}ms")

        plot_collective_curve(curve_dfs, p_label, length, out)
        plot_fitnesses(fitnesses, p_label, length, out)
        plot_evaluations_to_convergence(evaluations_to_converge, p_label, length, out)

    plot_times(times, length, out)
    return fitness_totals, times


def run():
    sns.set()
    sns.set_palette("Set2")

    ex1aOut = OUTPUT / "experiment1a/"
    upsert_directory(ex1aOut)
    run_with_length(100, ex1aOut)

    ex1bOut = OUTPUT / "experiment1b/"
    upsert_directory(ex1bOut)

    lengths = range(100, 1100, 100)
    fitnesses = []
    times = []
    for length in lengths:
        f, t = run_with_length(length, ex1bOut)
        fitnesses += [[length, *x] for x in f]
        times += [[length, *x] for x in t]

    fitness_df = pd.DataFrame(
        fitnesses, columns=["Length", "Problem", "Algorithm", "Fitness"]
    )
    time_df = pd.DataFrame(
        times, columns=["Length", "Problem", "Algorithm", "Time (ms)"]
    )

    print("----")
    plot_fitness_by_space(fitness_df, ex1bOut)
    plot_time_by_space(time_df, None)
