from src.models.qlearner import QLearner
from src.problems.lotr import (
    create as create_lotr,
    position_to_coords,
    actions,
    coords_to_position,
    get_possible_actions,
)
from src.experiments.plots import plot_lotr_policy
from src.util import round_

import numpy as np


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


def test_lotr():
    def has_won(s, r):
        return r > 0

    def has_lost(s, r):
        return r < -99

    n = 7
    P, R, W = create_lotr(n=n)
    model = QLearner(num_states=len(W.flatten()), num_actions=4)
    rewards = model.run(
        P, R, W, move_lotr, has_won, has_lost, n_epochs=1000, max_iters=1000
    )
    print(model.policy)
    print(rewards)

    plot_lotr_policy(
        np.array([round_(x) for x in model.policy]).reshape((n, n)),
        np.array([round_(x) for x in model.V]).reshape((n, n)),
        W,
        "lotr-policy-q.png",
        title="Small LOTR QL Policy and Utility",
    )


def run():
    test_lotr()
