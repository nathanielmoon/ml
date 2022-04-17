from extern.pymdptoolbox.src.mdptoolbox.mdp import PolicyIteration
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
from src.terms import PI_LABEL, PI_LABEL, QL_LABEL, LOTR_LABEL, FOREST_LABEL
from src.experiments.plots import (
    plot_iters_by_discount,
    plot_time_by_discount,
    plot_utilityreward_by_discount,
    plot_delta_by_iterations,
    plot_utilityreward_by_iterations,
    plot_lotr_policy,
    plot_forest_policy,
)


lotr_small = 7
lotr_large = 32
forest_small = 7
forest_large = 500

discounts = [x / 100 for x in range(60, 100, 2)]
problems = [
    (LOTR_LABEL, create_lotr, simulate_lotr, (lotr_small, lotr_large)),
    (FOREST_LABEL, create_forest, simulate_forest, (forest_small, forest_large)),
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
        model = PolicyIteration(P, R, discount, max_iter=1000)
        model.run()
        results.append(
            (
                discount,
                model.iter,
                model.time,
                simulate(model, P, R, W, max_steps=P.shape[1] * 2),
            )
        )
    return results


def run_iteratives(P, R, W, simulate):
    model = PolicyIteration(P, R, 0.95, max_iter=1000)
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
            results[plabel][n]["W"] = W
            results[plabel][n]["discount"] = run_discounts(P, R, W, simulate)
            (
                results[plabel][n]["iterations"],
                results[plabel][n]["model"],
            ) = run_iteratives(P, R, W, simulate)

    plots(results)


def plots(results):
    plot_lotr_policy(
        np.array(results[LOTR_LABEL][lotr_small]["model"].policy).reshape(
            (lotr_small, lotr_small)
        ),
        np.array(
            [round_(x) for x in results[LOTR_LABEL][lotr_small]["model"].V]
        ).reshape((lotr_small, lotr_small)),
        results[LOTR_LABEL][lotr_small]["W"],
        "pi-lotr-small-policy.png",
        title="Small LOTR PI Policy and Utility",
    )
    plot_forest_policy(
        np.array(results[FOREST_LABEL][forest_small]["model"].policy),
        np.array([round_(x) for x in results[FOREST_LABEL][forest_small]["model"].V]),
        results[FOREST_LABEL][forest_small]["R"],
        "pi-forest-small-policy.png",
        title="Small Forest PI Policy",
    )
    plot_lotr_policy(
        np.array(results[LOTR_LABEL][lotr_large]["model"].policy).reshape(
            (lotr_large, lotr_large)
        ),
        np.array(
            [round_(x) for x in results[LOTR_LABEL][lotr_large]["model"].V]
        ).reshape((lotr_large, lotr_large)),
        results[LOTR_LABEL][lotr_large]["W"],
        "pi-lotr-large-policy.png",
        title="Large LOTR PI Policy and Utility",
    )
    plot_forest_policy(
        np.array(results[FOREST_LABEL][forest_large]["model"].policy),
        np.array([round_(x) for x in results[FOREST_LABEL][forest_large]["model"].V]),
        results[FOREST_LABEL][forest_large]["R"],
        "pi-forest-large-policy.png",
        title="Large Forest PI Policy",
    )

    plot_iters_by_discount(
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        [x[1] for x in results[LOTR_LABEL][7]["discount"]],
        title="LOTR Small PI Iterations by Discount",
        filename="pi-lotr-small-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[1] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        title="LOTR Large PI Iterations by Discount",
        filename="pi-lotr-large-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][7]["discount"]],
        [x[1] for x in results[FOREST_LABEL][7]["discount"]],
        title="Forest Small PI Iterations by Discount",
        filename="pi-forest-small-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[1] for x in results[FOREST_LABEL][forest_large]["discount"]],
        title="Forest Large PI Iterations by Discount",
        filename="pi-forest-large-iters.png",
    )

    plot_time_by_discount(
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        [x[2] * 1000000 for x in results[LOTR_LABEL][7]["discount"]],
        title="LOTR Small PI Times by Discount",
        filename="pi-lotr-small-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[2] * 1000000 for x in results[LOTR_LABEL][lotr_large]["discount"]],
        title="LOTR Large PI Times by Discount",
        filename="pi-lotr-large-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[FOREST_LABEL][7]["discount"]],
        [x[2] * 1000000 for x in results[FOREST_LABEL][7]["discount"]],
        title="Forest Small PI Times by Discount",
        filename="pi-forest-small-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[2] * 1000000 for x in results[FOREST_LABEL][forest_large]["discount"]],
        title="Forest Large PI Times by Discount",
        filename="pi-forest-large-times.png",
    )

    plot_utilityreward_by_discount(
        [x[3][1] for x in results[LOTR_LABEL][7]["discount"]],
        [x[3][0] for x in results[LOTR_LABEL][7]["discount"]],
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        title="LOTR Small Reward by Discount",
        filename="pi-lotr-small-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[3][0] for x in results[LOTR_LABEL][lotr_large]["discount"]],
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        title="LOTR Large Reward by Discount",
        filename="pi-lotr-large-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][7]["discount"]],
        [x[3][0] for x in results[FOREST_LABEL][7]["discount"]],
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        title="Forest Small Reward by Discount",
        filename="pi-forest-small-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[3][0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[0] for x in results[LOTR_LABEL][7]["discount"]],
        title="Forest Large Reward by Discount",
        filename="pi-forest-large-discountxrewards.png",
    )

    plot_delta_by_iterations(
        [x[0] for x in results[LOTR_LABEL][7]["iterations"]],
        title="LOTR Small PI Delta by Iteration",
        filename="pi-lotr-small-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        title="LOTR Large PI Delta by Iteration",
        filename="pi-lotr-large-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[FOREST_LABEL][7]["iterations"]],
        title="LOTR Small PI Delta by Iteration",
        filename="pi-forest-small-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        title="LOTR Large PI Delta by Iteration",
        filename="pi-forest-large-deltaxiterations.png",
    )

    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[LOTR_LABEL][7]["iterations"]],
        [x[1][0] for x in results[LOTR_LABEL][7]["iterations"]],
        title="LOTR Small Reward by Iteration",
        filename="pi-lotr-small-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        [x[1][0] for x in results[LOTR_LABEL][lotr_large]["iterations"]],
        title="LOTR Large Reward by Iteration",
        filename="pi-lotr-large-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[FOREST_LABEL][7]["iterations"]],
        [x[1][0] for x in results[FOREST_LABEL][7]["iterations"]],
        title="Forest Small Reward by Iteration",
        filename="pi-forest-small-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        [x[1][0] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        title="Forest Large Reward by Iteration",
        filename="pi-forest-large-iterxrewards.png",
    )
