import mlrose_hiive as mlrose
import numpy as np


def generate(length=100):
    def maximizer(state):
        score = 0
        prev = -1
        for i in range(len(state)):
            val = state[i]
            if val != prev:
                score += 1
            prev = val
        return score

    fitness = mlrose.fitness.CustomFitness(maximizer)
    problem = mlrose.DiscreteOpt(
        length=length, fitness_fn=fitness, maximize=True, max_val=2
    )
    initial_state = np.random.random_integers(0, 1, (length,))

    return problem, initial_state
