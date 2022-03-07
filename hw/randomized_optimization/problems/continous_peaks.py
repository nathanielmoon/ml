import mlrose_hiive as mlrose
import numpy as np


def generate(length=100):
    fitness = mlrose.fitness.ContinuousPeaks()
    problem = mlrose.DiscreteOpt(
        length=length, fitness_fn=fitness, maximize=True, max_val=2
    )
    initial_state = np.random.random_integers(0, 1, (length,))

    return problem, initial_state
