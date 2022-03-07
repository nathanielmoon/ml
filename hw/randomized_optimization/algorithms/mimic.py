import mlrose_hiive as mlrose


def run(problem, init_state, max_attempts=1, max_iters=1000):
    best_state, best_fitness, curve = mlrose.mimic(
        problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        random_state=1,
        curve=True,
    )
    return best_state, best_fitness, curve
