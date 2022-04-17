import sys
import math

import numpy as np
from extern.pymdptoolbox.src.mdptoolbox.mdp import (
    PolicyIteration,
    ValueIteration,
    QLearning,
)
from src.experiments.plots import plot_lotr_policy

from src.util import pretty_print_policy

np.set_printoptions(threshold=sys.maxsize)


DEATH_REWARD = -100
WIN_REWARD = 5
tile_to_reward = [
    0,  # 0 - Normal tile
    DEATH_REWARD,  # 1 -  Death
    WIN_REWARD,  # 2 -  Objective
    -1,  # 3 - Danger
]

mount_doom = [
    [1, 0, 0, 1, 2],
    [1, 0, 0, 1, 0],
    [3, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

n_actions = 4
prob_success = 0.95


class actions:
    up = 0
    down = 1
    left = 2
    right = 3


action_map = [actions.up, actions.down, actions.left, actions.right]
readable_action_map = ["U", "D", "L", "R"]


def generate_world(n):
    world = np.zeros((n, n))
    world = np.random.choice(
        [x for x in range(len(tile_to_reward))], (n, n), p=[0.94, 0.02, 0.0, 0.04]
    )
    world[:7, -5:] = mount_doom
    world[:2, :2] = 0

    return world.astype("int32")


def display(world_):
    print(np.rot90(world_))


def position_to_coords(position, world):
    dimsize = world.shape[0]
    x = int(position / dimsize)
    y = position % dimsize
    return x, y


def coords_to_position(coords, world):
    dimsize = world.shape[0]
    x, y = coords
    return x * dimsize + y


def take_action(a, coords, w):
    x, y = coords
    dimsize = w.shape[0]
    if a == actions.up:
        if x <= 0:
            return -1
        return coords_to_position((x - 1, y), w)
    if a == actions.down:
        if x >= dimsize - 1:
            return -1
        return coords_to_position((x + 1, y), w)
    if a == actions.left:
        if y <= 0:
            return -1
        return coords_to_position((x, y - 1), w)
    if a == actions.right:
        if y >= dimsize - 1:
            return -1
        return coords_to_position((x, y + 1), w)


def get_possible_actions(coords, world_):
    return [take_action(a, coords, world_) for a in action_map]


def get_tile_probabilities(a, coords, world_):
    # RETURNS probabilies: stay, up, down, left, right
    possible_actions = get_possible_actions(coords, world_)

    # If target action is not possible, then stay
    if possible_actions[a] < 0:
        return (coords_to_position(coords, world_), 1.0), *[
            (x, 0.0) for x in possible_actions
        ]

    n_others = len([x for i, x in enumerate(possible_actions) if i != a and x > -1])
    action_probs = []
    for i in range(len(possible_actions)):
        if i == a:
            action_probs.append((possible_actions[i], prob_success))
        elif possible_actions[i] < 0:
            action_probs.append((possible_actions[i], 0.0))
        else:
            action_probs.append((possible_actions[i], (1.0 - prob_success) / n_others))

    return (coords_to_position(coords, world_), 0.0), *action_probs


def make_transition_matrix(world_):
    n_states = len(world_.flatten())
    matrix = np.zeros((n_actions, n_states, n_states))
    print(world_)

    for a in range(0, n_actions):
        for s in range(0, n_states):
            coords = position_to_coords(s, world_)
            probs = get_tile_probabilities(a, coords, world_)
            for pos, prob in probs:
                if pos < 0:
                    continue

                matrix[a][s][pos] = prob

    return matrix


def make_reward_matrix(world_):
    return np.array([tile_to_reward[x] for x in world_.flatten()])


def create(n=10):
    world = generate_world(n)
    P = make_transition_matrix(world)
    R = make_reward_matrix(world)
    return P, R, world


def simulate(Policy, P, R, world_, verbose=False, max_steps=1000):
    def has_died(p):
        return R[p] == DEATH_REWARD

    def has_won(p):
        return R[p] == WIN_REWARD

    def apply_move(pos, r):
        coords = position_to_coords(pos, world_)
        a = Policy[pos]
        stay, *action_probs = get_tile_probabilities(a, coords, world_)

        if stay[1] > 0:
            actual_a = np.random.choice(action_map)
        else:
            actual_a = np.random.choice(action_map, p=[x[1] for x in action_probs])
        next_pos = action_probs[actual_a][0]
        return next_pos, r + R[next_pos]

    reward = 0
    step = 0
    position = 0

    if verbose:
        print("Simulating ...")
    while not has_died(position) and not has_won(position) and step < max_steps:
        position, reward = apply_move(position, reward)
        step += 1
        if verbose:
            print(
                f"\tStep = {step}, Position = {position} - {position_to_coords(position, world_)}"
            )

    result = 0
    if has_died(position):
        result = 2
    elif has_won(position):
        result = 1
    else:
        result = 0

    if verbose:
        if has_died(position):
            print("You Died!")
        elif has_won(position):
            print("You Won!")
        else:
            print("Terminated.")
        print(
            f"Finished. Steps = {step}, Final Reward = {reward}, Final Coords = {position_to_coords(position, world_)}"
        )

    return reward, step, result


def compute_theoretical_reward_and_utility(
    model, P, R, W, max_steps=100, verbose=False, gamma=1
):
    def has_died(p):
        return R[p] == DEATH_REWARD

    def has_won(p):
        return R[p] == WIN_REWARD

    def apply_move(pos, r, u, step):
        coords = position_to_coords(pos, W)
        a = model.policy[pos]
        next_pos = take_action(a, coords, W)
        r_ = r + (R[next_pos] * math.pow(gamma, step))
        return next_pos, r_, u + model.V[next_pos]

    utility = 0
    reward = 0
    step = 0
    position = 0
    while not has_died(position) and not has_won(position) and step < max_steps:
        position, reward, utility = apply_move(position, reward, utility, step=step)
        step += 1
        if verbose:
            print(
                f"\tStep = {step}, Position = {position} - {position_to_coords(position, W)}, Reward = {reward}, Utility = {utility}"
            )

    return reward, utility


def run():
    P, R, world = create(n=7)

    print("WORLD")
    print(world)

    def hook(model, attr="V"):
        data = getattr(model, attr)
        return np.absolute(np.array(data)).sum() / 7

    vit = ValueIteration(P, R, 0.98, max_iter=1000000)
    signal = vit.run(hook=hook)
    print("Value Iteration Policy")
    print(pretty_print_policy(vit.policy, readable_action_map, dim_size=len(world)))
    reward, step, result = simulate(vit.policy, P, R, world, verbose=True)
    print("RESULT", reward, step, result)

    reward, utility = compute_theoretical_reward_and_utility(
        vit, P, R, world, verbose=True
    )

    print("THEORETICAL", reward, utility)

    return

    pit = PolicyIteration(P, R, 0.98, max_iter=100000)
    signal = pit.run(hook=hook)
    print("Policy Iteration Policy")
    print(pretty_print_policy(pit.policy, readable_action_map, dim_size=len(world)))
    print(pit.V)

    plot_lotr_policy(
        np.array(pit.policy).reshape((7, 7)),
        world,
        "tmp.png",
    )

    print("CONFIDENCE:", np.absolute(np.array(pit.V)).sum() / 7)
    print(signal)

    reward, step, result = simulate(pit.policy, P, R, world, verbose=False)
    print("RESULT", reward, step, result)

    ql = QLearning(P, R, 1.0, n_iter=1000000)
    ql.run()
    print("Q-Learning Policy")
    print(pretty_print_policy(ql.policy, readable_action_map, dim_size=len(world)))
    reward, step, result = simulate(ql.policy, P, R, world, verbose=False)
    print("RESULT", reward, step, result)
