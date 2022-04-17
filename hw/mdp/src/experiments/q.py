import time

from extern.pymdptoolbox.src.mdptoolbox.mdp import QLearning
from src.models.qlearner import QLearner
import numpy as np
import seaborn as sns

from src.problems.forest import (
    create as create_forest,
    compute_theoretical_reward_and_utility as simulate_forest,
)
from src.problems.lotr import (
    create as create_lotr,
    compute_theoretical_reward_and_utility as simulate_lotr,
    position_to_coords,
    get_possible_actions,
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


def pass_through(s, r):
    return False


def move_forest(s, a, P, R, W):
    p = np.random.random()
    n = P.shape[1]

    if p > 0.9:
        return 0, 0

    if a == 0:
        if s == n - 1:
            return s, R[s][a]
        else:
            return s + 1, R[s][a]

    else:
        if s == n - 1:
            return 0, R[s][a]
        else:
            return 0, R[s][a]


def move_lotr(s, a, P, R, W):
    x, y = position_to_coords(s, W)
    possible_actions = get_possible_actions((x, y), W)
    num_actions = len([x for x in possible_actions if x > -1])

    if possible_actions[a] > -1:
        p = [
            (1.0 - 0.95) / (num_actions - 1) if i != a and x > -1 else 0.0
            for i, x in enumerate(possible_actions)
        ]
        p[a] = 0.95
        s_ = np.random.choice(possible_actions, p=p)
    else:
        p = [
            1.0 / num_actions if i != a and x > -1 else 0.0
            for i, x in enumerate(possible_actions)
        ]
        s_ = np.random.choice(possible_actions, p=p)

    return s_, R[s_]


lotr_small = 7
lotr_large = 32
forest_small = 7
forest_large = 500
n_iter = 20000
alpha = 0.3

discounts = [x / 100 for x in range(60, 100, 2)]
alphas = [x / 10 for x in range(1, 10)]
problems = [
    (LOTR_LABEL, create_lotr, simulate_lotr, (lotr_small,), move_lotr),
    (
        FOREST_LABEL,
        create_forest,
        simulate_forest,
        (forest_small, forest_large),
        move_forest,
    ),
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
    gamma=1,
):
    sim = None
    if simulate:
        sim = simulate(model, P, R, W, verbose=False, max_steps=max_steps, gamma=gamma)

    return (
        variation,
        sim,
    )


def run_discounts(P, R, W, simulate, move):
    results = []
    n = P.shape[1]
    for discount in discounts:
        start = time.time()
        model = QLearner(num_states=n, num_actions=P.shape[0], gamma=discount)
        model.run(
            P,
            R,
            pass_through,
            pass_through,
            n_epochs=500,
            max_iters=n * 2,
        )
        end = time.time()
        results.append(
            (
                discount,
                0,
                end - start,
                simulate(model, P, R, W, max_steps=P.shape[1] * 2, gamma=discount),
            )
        )
    return results


def run_alphas(P, R, W, simulate, move):
    results = []
    n = P.shape[1]
    gamma = 0.98
    for alpha in alphas:
        start = time.time()
        model = QLearner(num_states=n, num_actions=P.shape[0], gamma=gamma, alpha=alpha)
        model.run(
            P,
            R,
            pass_through,
            pass_through,
            n_epochs=500,
            max_iters=n * 2,
        )
        end = time.time()
        results.append(
            (
                alpha,
                0,
                end - start,
                simulate(model, P, R, W, max_steps=P.shape[1] * 2, gamma=gamma),
            )
        )
    return results


def run_iteratives(P, R, W, simulate, move):
    n = P.shape[1]
    gamma = 0.98
    model = QLearner(num_states=n, num_actions=P.shape[0], gamma=gamma)
    signals = model.run(
        P,
        R,
        pass_through,
        pass_through,
        n_epochs=500,
        max_iters=n * 2,
        hook=hook,
        W=W,
        simulate=simulate,
    )
    return signals, model


def run():
    sns.set()
    upsert_directory(OUTPUT_DIR)
    clear_cache(CACHE_DIR)
    np.random.seed(42)

    results = {}
    for plabel, create, simulate, sizes, move in problems:
        results[plabel] = {}
        for n in sizes:
            results[plabel][n] = {}
            print(plabel, n)
            P, R, W = create(n)
            results[plabel][n]["P"] = P
            results[plabel][n]["R"] = R
            results[plabel][n]["W"] = W
            results[plabel][n]["discount"] = run_discounts(P, R, W, simulate, move)
            results[plabel][n]["alphas"] = run_alphas(P, R, W, simulate, move)
            (
                results[plabel][n]["iterations"],
                results[plabel][n]["model"],
            ) = run_iteratives(P, R, W, simulate, move)

    plots(results)
    print("\a")


def plots(results):
    plot_lotr_policy(
        np.array(results[LOTR_LABEL][lotr_small]["model"].policy).reshape(
            (lotr_small, lotr_small)
        ),
        np.array(
            [round_(x) for x in results[LOTR_LABEL][lotr_small]["model"].V]
        ).reshape((lotr_small, lotr_small)),
        results[LOTR_LABEL][lotr_small]["W"],
        "ql-lotr-small-policy.png",
        title="Small LOTR QL Policy and Utility",
    )
    plot_forest_policy(
        np.array(results[FOREST_LABEL][forest_small]["model"].policy),
        np.array([round_(x) for x in results[FOREST_LABEL][forest_small]["model"].V]),
        results[FOREST_LABEL][forest_small]["R"],
        "ql-forest-small-policy.png",
        title="Small Forest QL Policy",
    )
    plot_forest_policy(
        np.array(results[FOREST_LABEL][forest_large]["model"].policy),
        np.array([round_(x) for x in results[FOREST_LABEL][forest_large]["model"].V]),
        results[FOREST_LABEL][forest_large]["R"],
        "ql-forest-large-policy.png",
        title="Large Forest QL Policy",
    )

    plot_iters_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        [x[1] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        title="LOTR Small QL Iterations by Discount",
        filename="ql-lotr-small-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[1] for x in results[FOREST_LABEL][forest_small]["discount"]],
        title="Forest Small QL Iterations by Discount",
        filename="ql-forest-small-iters.png",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[1] for x in results[FOREST_LABEL][forest_large]["discount"]],
        title="Forest Large QL Iterations by Discount",
        filename="ql-forest-large-iters.png",
    )

    plot_time_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        [x[2] * 1000000 for x in results[LOTR_LABEL][lotr_small]["discount"]],
        title="LOTR Small QL Times by Discount",
        filename="ql-lotr-small-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[2] * 1000000 for x in results[FOREST_LABEL][forest_small]["discount"]],
        title="Forest Small QL Times by Discount",
        filename="ql-forest-small-times.png",
    )
    plot_time_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[2] * 1000000 for x in results[FOREST_LABEL][forest_large]["discount"]],
        title="Forest Large QL Times by Discount",
        filename="ql-forest-large-times.png",
    )

    plot_utilityreward_by_discount(
        [x[3][1] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        [x[3][0] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        [x[0] for x in results[LOTR_LABEL][lotr_small]["discount"]],
        title="LOTR Small Reward by Discount",
        filename="ql-lotr-small-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[3][0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        [x[0] for x in results[FOREST_LABEL][forest_small]["discount"]],
        title="Forest Small Reward by Discount",
        filename="ql-forest-small-discountxrewards.png",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[3][0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        [x[0] for x in results[FOREST_LABEL][forest_large]["discount"]],
        title="Forest Large Reward by Discount",
        filename="ql-forest-large-discountxrewards.png",
    )

    plot_utilityreward_by_discount(
        [x[3][1] for x in results[LOTR_LABEL][lotr_small]["alphas"]],
        [x[3][0] for x in results[LOTR_LABEL][lotr_small]["alphas"]],
        [x[0] for x in results[LOTR_LABEL][lotr_small]["alphas"]],
        title="LOTR Small Reward by Alpha",
        filename="ql-lotr-small-alphaxrewards.png",
        xlabel="Alpha",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][forest_small]["alphas"]],
        [x[3][0] for x in results[FOREST_LABEL][forest_small]["alphas"]],
        [x[0] for x in results[FOREST_LABEL][forest_small]["alphas"]],
        title="Forest Small Reward by Alpha",
        filename="ql-forest-small-alphaxrewards.png",
        xlabel="Alpha",
    )
    plot_utilityreward_by_discount(
        [x[3][1] for x in results[FOREST_LABEL][forest_large]["alphas"]],
        [x[3][0] for x in results[FOREST_LABEL][forest_large]["alphas"]],
        [x[0] for x in results[FOREST_LABEL][forest_large]["alphas"]],
        title="Forest Large Reward by Alpha",
        filename="ql-forest-large-alphaxrewards.png",
        xlabel="Alpha",
    )

    plot_iters_by_discount(
        [x[0] for x in results[LOTR_LABEL][lotr_small]["alphas"]],
        [x[1] for x in results[LOTR_LABEL][lotr_small]["alphas"]],
        title="LOTR Small QL Iterations by Alpha",
        filename="ql-lotr-small-itersxalpha.png",
        xlabel="Alpha",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_small]["alphas"]],
        [x[1] for x in results[FOREST_LABEL][forest_small]["alphas"]],
        title="Forest Small QL Iterations by Alpha",
        filename="ql-forest-small-itersxalpha.png",
        xlabel="Alpha",
    )
    plot_iters_by_discount(
        [x[0] for x in results[FOREST_LABEL][forest_large]["alphas"]],
        [x[1] for x in results[FOREST_LABEL][forest_large]["alphas"]],
        title="Forest Large QL Iterations by Alpha",
        filename="ql-forest-large-itersxalpha.png",
        xlabel="Alpha",
    )

    plot_delta_by_iterations(
        [x[0] for x in results[LOTR_LABEL][7]["iterations"]],
        title="LOTR Small QL Delta by Iteration",
        filename="ql-lotr-small-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[FOREST_LABEL][forest_small]["iterations"]],
        title="Forest Small QL Delta by Iteration",
        filename="ql-forest-small-deltaxiterations.png",
    )
    plot_delta_by_iterations(
        [x[0] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        title="Forest Large QL Delta by Iteration",
        filename="ql-forest-large-deltaxiterations.png",
    )

    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[LOTR_LABEL][lotr_small]["iterations"]],
        [x[1][0] for x in results[LOTR_LABEL][lotr_small]["iterations"]],
        title="LOTR Small Reward by Iteration",
        filename="ql-lotr-small-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[FOREST_LABEL][forest_small]["iterations"]],
        [x[1][0] for x in results[FOREST_LABEL][forest_small]["iterations"]],
        title="Forest Small Reward by Iteration",
        filename="ql-forest-small-iterxrewards.png",
    )
    plot_utilityreward_by_iterations(
        [x[1][1] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        [x[1][0] for x in results[FOREST_LABEL][forest_large]["iterations"]],
        title="Forest Large Reward by Iteration",
        filename="ql-forest-large-iterxrewards.png",
    )
