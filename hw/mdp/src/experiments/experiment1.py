# TODO
#   Write simulator for forest problem
#   Write policy, value, and Q runs and data collect
#   Vary epsilon
#   Watch office hours

from cgitb import small
import json

from extern.pymdptoolbox.src.mdptoolbox.mdp import (
    PolicyIteration,
    ValueIteration,
    QLearning,
)
import numpy as np
import seaborn as sns
from scipy.stats import kurtosis

from src.problems.forest import (
    create as create_forest,
    compute_theoretical_reward_and_utility as simulate_forest,
)
from src.problems.lotr import (
    create as create_lotr,
    compute_theoretical_reward_and_utility as simulate_lotr,
)
from src.util import clear_cache, upsert_directory, write_cache
from src.paths import CACHE_DIR, OUTPUT_DIR
from src.terms import PI_LABEL, VI_LABEL, QL_LABEL, LOTR_LABEL, FOREST_LABEL
from src.experiments.plots import (
    plot_lotr_world,
    plot_iterative_curve,
    plot_lotr_policy,
    plot_utilityreward,
    plot_forest_policy,
)

problems = [
    (LOTR_LABEL, create_lotr, simulate_lotr),
    (FOREST_LABEL, create_forest, simulate_forest),
]

algorithms = [
    (PI_LABEL, PolicyIteration, {"max_iter": 1000}),
    (VI_LABEL, ValueIteration, {"max_iter": 1000}),
    # (QL_LABEL, QLearning, {"n_iter": 20000}),
]

lotr_small_problem_size = 7
lotr_large_problem_size = 20
sizes = [lotr_small_problem_size, lotr_large_problem_size]
epsilons = [x / 100 for x in range(60, 100, 2)]
n_sims = 100


def compute_confidence(model, attr="V"):
    data = getattr(model, attr)
    return np.absolute(np.array(data)).mean()


def hook(model, attr="V", simulate=None, P=None, R=None, W=None):
    confidence = compute_confidence(model, attr=attr)

    sim = None
    if simulate:
        sim = run_simulation(simulate, model, P, R, W, n=10)

    return (
        confidence,
        sim,
    )


def run_simulation(simulate, model, P, R, W, n=100):
    return simulate(model, P, R, W, verbose=False)


def run_epsilons(P, R, W, n, ctor, simulate, plabel, alabel, params):
    rewards = []
    policies = []
    times = []
    train_iterations = []
    confidences = []
    utilities = []
    for eidx, epsilon in enumerate(epsilons):
        print(f"\te={epsilon}")
        model = ctor(P, R, epsilon, **params)
        signal = model.run()
        reward, utility = run_simulation(simulate, model, P, R, W)

        utilities.append(utility)
        rewards.append(reward)
        policies.append(model.policy)
        times.append(model.time)
        train_iterations.append(None if alabel == QL_LABEL else model.iter)

        if signal:
            confidences.append(signal[0])
        write_cache(
            CACHE_DIR,
            f"{plabel}-{alabel}-{n}-e-{epsilon}.json",
            json.dumps(model.policy, indent=2),
        )

    return {
        "rewards": np.array(rewards),
        "policies": np.array(policies),
        "times": np.array(times),
        "train_iterations": np.array(train_iterations),
        "confidences": np.array(confidences),
        "utilities": utilities,
    }


def run_iterations(epsilon, P, R, W, n, ctor, simulate, plabel, alabel, params):
    policies = []
    times = []
    model = ctor(
        P,
        R,
        epsilon,
        **params,
    )
    signal = model.run(hook=hook, params={"P": P, "R": R, "W": W, "simulate": simulate})
    rewards = [x[1][0] for x in signal]
    utilities = [x[1][1] for x in signal]

    write_cache(
        CACHE_DIR,
        f"{plabel}-{alabel}-{n}-iterations.json",
        json.dumps(model.policy, indent=2),
    )

    return {
        "rewards": rewards,
        "policy": np.array(model.policy),
        "times": times,
        "confidences": [x[0] for x in signal],
        "utilities": utilities,
    }


def run_experiment():
    sns.set()
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
    plot_lotr_world(results[LOTR_LABEL][lotr_small_problem_size]["W"])
    plot_lotr_policy(
        results[LOTR_LABEL][lotr_small_problem_size][PI_LABEL]["epsilons"]["policies"][
            -1
        ].reshape((lotr_small_problem_size, lotr_small_problem_size)),
        results[LOTR_LABEL][lotr_small_problem_size]["W"],
        "lotr-small-pi-policy.png",
        title="Small LOTR Policy Iteration Policy",
    )
    plot_lotr_policy(
        results[LOTR_LABEL][lotr_small_problem_size][VI_LABEL]["epsilons"]["policies"][
            -1
        ].reshape((lotr_small_problem_size, lotr_small_problem_size)),
        results[LOTR_LABEL][lotr_small_problem_size]["W"],
        "lotr-small-vi-policy.png",
        title="Small LOTR Value Iteration Policy",
    )

    """ ITERATION CONFIDENCES """
    plot_utilityreward(
        results[LOTR_LABEL][lotr_small_problem_size][VI_LABEL]["iterations"][
            "utilities"
        ],
        results[LOTR_LABEL][lotr_small_problem_size][VI_LABEL]["iterations"]["rewards"],
        "LOTR Small VI Utility/Reward Curve",
        "lotr-small-vi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[FOREST_LABEL][lotr_small_problem_size][VI_LABEL]["iterations"][
            "utilities"
        ],
        results[FOREST_LABEL][lotr_small_problem_size][VI_LABEL]["iterations"][
            "rewards"
        ],
        "Forest Small VI Utility/Reward Curve",
        "forest-small-vi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[LOTR_LABEL][lotr_small_problem_size][PI_LABEL]["iterations"][
            "utilities"
        ],
        results[LOTR_LABEL][lotr_small_problem_size][PI_LABEL]["iterations"]["rewards"],
        "LOTR Small PI Utility/Reward Curve",
        "lotr-small-pi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[FOREST_LABEL][lotr_small_problem_size][PI_LABEL]["iterations"][
            "utilities"
        ],
        results[FOREST_LABEL][lotr_small_problem_size][PI_LABEL]["iterations"][
            "rewards"
        ],
        "Forest Small PI Utility/Reward Cruve",
        "forest-small-pi-iteration-utilityreward.png",
    )

    plot_utilityreward(
        results[LOTR_LABEL][lotr_large_problem_size][VI_LABEL]["iterations"][
            "utilities"
        ],
        results[LOTR_LABEL][lotr_large_problem_size][VI_LABEL]["iterations"]["rewards"],
        "LOTR Large VI Utility/Reward Curve",
        "lotr-large-vi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[FOREST_LABEL][lotr_large_problem_size][VI_LABEL]["iterations"][
            "utilities"
        ],
        results[FOREST_LABEL][lotr_large_problem_size][VI_LABEL]["iterations"][
            "rewards"
        ],
        "Forest Large VI Utility/Reward Curve",
        "forest-large-vi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[LOTR_LABEL][lotr_large_problem_size][PI_LABEL]["iterations"][
            "utilities"
        ],
        results[LOTR_LABEL][lotr_large_problem_size][PI_LABEL]["iterations"]["rewards"],
        "LOTR Large PI Utility/Reward Curve",
        "lotr-large-pi-iteration-utilityreward.png",
    )
    plot_utilityreward(
        results[FOREST_LABEL][lotr_large_problem_size][PI_LABEL]["iterations"][
            "utilities"
        ],
        results[FOREST_LABEL][lotr_large_problem_size][PI_LABEL]["iterations"][
            "rewards"
        ],
        "Forest Large PI Utility/Reward Cruve",
        "forest-large-pi-iteration-utilityreward.png",
    )


def run():
    results = run_experiment()
    plots(results)
