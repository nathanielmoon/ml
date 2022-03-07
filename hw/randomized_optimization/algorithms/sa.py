import mlrose_hiive as mlrose


def run(problem, init_state, max_attempts=1, max_iters=1000):
    schedule = mlrose.ExpDecay()
    best_state, best_fitness, curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=max_attempts,
        max_iters=max_iters,
        init_state=init_state,
        random_state=1,
        curve=True,
    )
    return best_state, best_fitness, curve
