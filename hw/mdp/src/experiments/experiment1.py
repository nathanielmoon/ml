# TODO
#   Write simulator for forest problem
#   Write policy, value, and Q runs and data collect
#   Vary epsilon
#   Watch office hours

import json

from mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import numpy as np

from src.problems.forest import create as create_forest, simulate as simulate_forest
from src.problems.lotr import create as create_lotr, simulate as simulate_lotr
from src.util import clear_cache, upsert_directory, write_cache
from src.paths import CACHE_DIR, OUTPUT_DIR
from src.terms import PI_LABEL, VI_LABEL, QL_LABEL, LOTR_LABEL, FOREST_LABEL
from src.experiments.plots import (
    plot_lotr_world,
    plot_iterative_curve,
    plot_lotr_policy,
)

problems = [
    (LOTR_LABEL, create_lotr, simulate_lotr),
    (FOREST_LABEL, create_forest, simulate_forest),
]

algorithms = [
    (PI_LABEL, PolicyIteration, {"max_iter": 1000}),
    (VI_LABEL, ValueIteration, {"max_iter": 1000}),
    (QL_LABEL, QLearning, {"n_iter": 100001}),
]

sizes = [10]  # , 32]
epsilons = [x / 100 for x in range(80, 100, 1)]
iterationMap = {
    10: {
        PI_LABEL: list(range(10, 1110, 100)),
        VI_LABEL: list(range(10, 1110, 100)),
        QL_LABEL: list(range(10000, 21000, 1000)),
    },
    32: {
        PI_LABEL: list(range(1000, 2000, 100)),
        VI_LABEL: list(range(1000, 2000, 100)),
        QL_LABEL: list(range(10000, 21000, 1000)),
    },
}
n_sims = 100


def run_simulation(simulate, model, P, R, W):
    rewards, iterations, signals = np.zeros(n_sims), np.zeros(n_sims), np.zeros(n_sims)
    for i in range(n_sims):
        reward, step, signal = simulate(model.policy, P, R, W)
        rewards[i] = reward
        iterations[i] = step
        signals[i] = signal
    return rewards.tolist(), iterations.tolist(), signals.tolist()


def run_epsilons(P, R, W, n, ctor, simulate, plabel, alabel, params):
    totalRewards = []
    policies = []
    times = []
    sim_iterations = []
    train_iterations = []
    for eidx, epsilon in enumerate(epsilons):
        print(f"\te={epsilon}")
        model = ctor(P, R, epsilon, **params)
        model.run()
        rewards, iters, signals = run_simulation(simulate, model, P, R, W)

        totalRewards.append(rewards)
        policies.append(model.policy)
        times.append(model.time)
        train_iterations.append(None if alabel == QL_LABEL else model.iter)
        sim_iterations.append(iters)
        write_cache(
            CACHE_DIR,
            f"{plabel}-{alabel}-{n}-e-{epsilon}.json",
            json.dumps(model.policy, indent=2),
        )

    return {
        "rewards": np.array(totalRewards),
        "policies": np.array(policies),
        "times": np.array(times),
        "sim_iterations": np.array(sim_iterations),
        "train_iterations": np.array(train_iterations),
    }


def run_iterations(epsilon, P, R, W, n, ctor, simulate, plabel, alabel, params):
    iterations = iterationMap[n][alabel]

    totalRewards = []
    policies = []
    times = []
    sim_iterations = []
    train_iterations = []
    for iidx, iteration in enumerate(iterations):
        print(f"\ti={iteration}")
        model = ctor(
            P,
            R,
            epsilon,
            **{**params, "n_iter" if alabel == QL_LABEL else "max_iter": iteration},
        )
        model.run()
        rewards, iters, signals = run_simulation(simulate, model, P, R, W)

        totalRewards.append(rewards)
        policies.append(model.policy)
        times.append(model.time)
        train_iterations.append(None if alabel == QL_LABEL else model.iter)
        sim_iterations.append(iters)
        write_cache(
            CACHE_DIR,
            f"{plabel}-{alabel}-{n}-i-{iteration}.json",
            json.dumps(model.policy, indent=2),
        )

    return {
        "rewards": np.array(totalRewards),
        "policies": np.array(policies),
        "times": np.array(times),
        "sim_iterations": np.array(sim_iterations),
        "train_iterations": np.array(train_iterations),
    }


def run_experiment():
    upsert_directory(OUTPUT_DIR)
    clear_cache(CACHE_DIR)
    np.random.seed(42)

    results = {
        p: {
            n: {
                a: {
                    "epsilons": None,
                    "iterations": None,
                }
                for a, _, _ in algorithms
            }
            for n in sizes
        }
        for p, _, _ in problems
    }
    for plabel, create, simulate in problems:
        for nidx, n in enumerate(sizes):
            P, R, W = create(n)
            results[plabel][n]["P"] = P
            results[plabel][n]["R"] = R
            results[plabel][n]["W"] = W

            for alabel, ctor, params in algorithms:
                print(f"{plabel}-{n}-{alabel}")
                results[plabel][n][alabel]["epsilons"] = run_epsilons(
                    P, R, W, n, ctor, simulate, plabel, alabel, params
                )
                results[plabel][n][alabel]["iterations"] = run_iterations(
                    0.99, P, R, W, n, ctor, simulate, plabel, alabel, params
                )

    return results


def plots(results):
    plot_lotr_world(results[LOTR_LABEL][10]["W"])
    plot_iterative_curve(
        epsilons,
        results[LOTR_LABEL][10][VI_LABEL]["epsilons"]["rewards"],
        ylabel="Reward",
        xlabel="Epsilon",
        title="Small LOTR Epsilon Curve for Value Iteration",
        filename="lotr-small-vi-epsilon.png",
    )
    plot_lotr_policy(
        results[LOTR_LABEL][10]["policies"][-1], results[LOTR_LABEL][10]["W"]
    )


def run():
    results = run_experiment()
    plots(results)
