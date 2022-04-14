from tabnanny import verbose
import mdptoolbox.example
from mdptoolbox.mdp import PolicyIteration
import numpy as np

sim_years = 100


def create(n=10):
    _n = n * n
    r1 = _n * 3
    r2 = _n * 2
    p = 1.0 / _n * 4
    P, R = mdptoolbox.example.forest(S=_n, r1=r1, r2=r2, p=p)
    return P, R, None


def simulate(Policy, P, R, world_, verbose=False):
    def apply_move(pos, r):
        a = Policy[pos]
        s_s = P[a][pos]
        s_ = np.random.choice([x for x in range(len(s_s))], p=s_s)
        return s_, r + R[pos][a]

    reward = 0
    max_steps = 1000
    step = 0
    position = 0

    if verbose:
        print("Simulating ...")
    while step < max_steps:
        position, reward = apply_move(position, reward)
        step += 1
        if verbose:
            print(f"\tStep = {step}, Position = {position}")

    if verbose:
        print(
            f"Finished. Steps = {step}, Final Reward = {reward}, Final Coords = {position}"
        )

    return reward, step, position


def run():
    P, R, _ = create()

    pot = PolicyIteration(P, R, 0.98, max_iter=len(R) * 2)
    pot.run()

    simulate(pot.policy, P, R, None, verbose=True)
    print(pot.policy)
