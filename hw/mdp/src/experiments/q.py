import time

from extern.pymdptoolbox.src.mdptoolbox.mdp import QLearning
import numpy as np
import seaborn as sns

from src.problems.forest import (
    create as create_forest,
    compute_theoretical_reward_and_utility as simulate_forest,
)
from src.problems.lotr import (
    create as create_lotr,
    compute_theoretical_reward_and_utility as simulate_lotr,
)
from src.util import clear_cache, upsert_directory, write_cache, round_
from src.paths import CACHE_DIR, OUTPUT_DIR
from src.terms import PI_LABEL, QL_LABEL, QL_LABEL, LOTR_LABEL, FOREST_LABEL
from src.experiments.plots import (
    plot_iters_by_discount,
    plot_time_by_discount,
    plot_utilityreward_by_discount,
    plot_delta_by_iterations,
    plot_utilityreward_by_iterations,
    plot_lotr_policy,
    plot_forest_policy,
)

"""
TODO
    Reward Vs Steps
    Delta Vs Steps

    Iterations vs Discount      <
    Time vs Discount            <
    Total Utility vs Discount   <

"""

lotr_small = 7
lotr_large = 32
forest_small = 7
forest_large = 500
n_iter = 100000

discounts = [x / 100 for x in range(60, 100, 2)]
problems = [
    (LOTR_LABEL, create_lotr, simulate_lotr, (lotr_large,)),
    (FOREST_LABEL, create_forest, simulate_forest, (forest_small,)),
]


def hook(
    model,
    attr="V",
    simulate=None,
    P=None,
    R=None,
    W=None,
    variation=None,
    max_steps=100,
):
    sim = None
    if simulate:
        sim = simulate(model, P, R, W, verbose=False, max_steps=max_steps)

    return (
        variation,
        sim,
    )


def run_discounts(P, R, W, simulate):
    results = []
    for discount in discounts:
        model = QLearning(P, R, discount, n_iter=n_iter)
        start = time.time()
        model.run()
        end = time.time()
        results.append(
            (
                discount,
                n_iter,
                end - start,
                simulate(model, P, R, W, max_steps=P.shape[1] * 2),
            )
        )
    return results


def run_iteratives(P, R, W, simulate):
    model = QLearning(P, R, 0.95, n_iter=n_iter)
    signal = model.run(
        hook=hook,
        params={
            "P": P,
            "R": R,
            "W": W,
            "simulate": simulate,
            "max_steps": P.shape[1] * 2,
        },
    )
    return signal, model


def run():
    sns.set()
    upsert_directory(OUTPUT_DIR)
    clear_cache(CACHE_DIR)
    np.random.seed(42)

    results = {}
    for plabel, create, simulate, sizes in problems:
        results[plabel] = {}
        for n in sizes:
            results[plabel][n] = {}
            print(plabel, n)
            P, R, W = create(n)
            results[plabel][n]["P"] = P
            results[plabel][n]["R"] = R
            results[plabel][n]["W"] = W
            results[plabel][n]["discount"] = run_discounts(P, R, W, simulate)
            (
                results[plabel][n]["iterations"],
                results[plabel][n]["model"],
            ) = run_iteratives(P, R, W, simulate)

    plots(results)


def plots(results):
    plot_forest_policy(
        np.array(results[FOREST_LABEL][forest_small]["model"].policy),
        np.array([round_(x) for x in results[FOREST_LABEL][forest_small]["model"].V]),
        results[FOREST_LABEL][forest_small]["R"],
        "ql-forest-small-policy.png",
        title="Small Forest QL Policy",
    )
    plot_lotr_policy(
        np.array(results[LOTR_LABEL][lotr_large]["model"].policy).reshape(
            (lotr_large, lotr_large)
        ),
        np.array(
            [round_(x) for x in results[LOTR_LABEL][lotr_large]["model"].V]
        ).reshape((lotr_large, lotr_large)),
        results[LOTR_LABEL][lotr_large]["W"],
        "ql-lotr-large-policy.png",
        title="Large LOTR QL Policy and Utility",
    )

    plot_iters_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[1] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        title="LOTR Large QL Iterations by Discount",
        filename="ql-lotr-large-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[1] for x in results[FOREST_LABEL][forest_small]["discount"]],
        title="Forest Small QL Iterations by Discount",
        filename="ql-forest-small-iters.png",
    )

    plot_time_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[2] * 1000000 for x in results[LOTR_LABEL][lotr_large]["discount"]],
        title="LOTR Large QL Times by Discount",
        filename="ql-lotr-large-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[2] * 1000000 for x in results[FOREST_LABEL][forest_small]["discount"]],
        title="Forest Small QL Times by Discount",
        filename="ql-forest-small-times.png",
    )

    plot_utilityreward_by_discount(
        [x[3][1] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[3][0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        title="LOTR Large Reward by Discount",
        filename="ql-lotr-large-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][7]["discount"]],
        [x[3][0] for x in results[FOREST_LABEL][7]["discount"]],
        [x[0] for x in results[FOREST_LABEL][7]["discount"]],
        title="Forest Small Reward by Discount",
        filename="ql-forest-small-discountxrewards.png",
    )

    plot_delta_by_iterations(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        title="LOTR Large QL Delta by Iteration",
        filename="ql-lotr-large-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[FOREST_LABEL][7]["iterations"]],
        title="LOTR Small QL Delta by Iteration",
        filename="ql-forest-small-deltaxiterations.png",
    )

    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        [x[1][0] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        title="LOTR Large Reward by Iteration",
        filename="ql-lotr-large-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[FOREST_LABEL][7]["iterations"]],
        [x[1][0] for x in results[FOREST_LABEL][7]["iterations"]],
        title="Forest Small Reward by Iteration",
        filename="ql-forest-small-iterxrewards.png",
    )
