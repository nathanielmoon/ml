import math
import mdptoolbox.example
from extern.pymdptoolbox.src.mdptoolbox.mdp import (
    PolicyIteration,
    ValueIteration,
    QLearning,
)
import numpy as np

sim_years = 100


def create(n=3):
    _n = n
    r1 = _n + 10
    r2 = _n * 1000000
    p = 0.1
    P, R = mdptoolbox.example.forest(S=_n, r1=r1, r2=r2, p=p)
    return P, R, None


def simulate(Policy, P, R, world_, verbose=False, max_steps=100):
    def apply_move(pos, r):
        a = Policy[pos]
        s_s = P[a][pos]
        s_ = np.random.choice([x for x in range(len(s_s))], p=s_s)
        return s_, r + R[s_][a]

    reward = 0
    step = 0
    position = 0

    if verbose:
        print("Simulating ...")
    while step < max_steps:
        position, reward = apply_move(position, reward)
        step += 1
        if verbose:
            print(f"\tStep = {step}, Position = {position}, Reward = {reward}")

    if verbose:
        print(
            f"Finished. Steps = {step}, Final Reward = {reward}, Final Coords = {position}"
        )

    return reward, step, position


def compute_theoretical_reward_and_utility(
    model, P, R, W, max_steps=100, verbose=False, gamma=1
):
    def apply_move(pos, r, u, step):
        a = model.policy[pos]
        s_s = P[a][pos]
        s_ = np.argmax(s_s)
        r_ = r + (R[pos][a] * math.pow(gamma, step))
        return s_, r_, u + model.V[s_]

    reward = 0
    utility = 0
    step = 0
    position = 0

    if verbose:
        print("Simulating ...")
    while step < max_steps:
        position, reward, utility = apply_move(position, reward, utility, step)
        step += 1
        if verbose:
            print(f"\tStep = {step}, Position = {position}, Reward = {reward}")

    if verbose:
        print(
            f"Finished. Steps = {step}, Final Reward = {reward}, Final Coords = {position}"
        )

    return reward, utility


def run():
    n = 7
    P, R, world = create(n=n)

    print("WORLD")
    print(world)

    def hook(model, attr="V", variation=None):
        data = getattr(model, attr)
        return np.absolute(np.array(data)).sum() / n

    vit = ValueIteration(P, R, 0.98, max_iter=1000)
    signal = vit.run(hook=hook)
    print("Value Iteration Policy")
    reward, utility = compute_theoretical_reward_and_utility(
        vit, P, R, world, verbose=True, max_steps=n * 2
    )
    print("THEORITCAL", reward, utility)
    print(vit.policy)
    return

    pit = PolicyIteration(P, R, 0.98, max_iter=100000)
    signal = pit.run(hook=hook)
    print("Policy Iteration Policy")
    print(pit.V)

    print("CONFIDENCE:", np.absolute(np.array(pit.V)).sum() / n)
    print(signal)

    reward, step, result = simulate(pit.policy, P, R, world, verbose=False)
    print("RESULT", reward, step, result)

    ql = QLearning(P, R, 1.0, n_iter=1000000)
    ql.run()
    print("Q-Learning Policy")
    reward, step, result = simulate(ql.policy, P, R, world, verbose=False)
    print("RESULT", reward, step, result)
