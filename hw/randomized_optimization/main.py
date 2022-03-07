import mlrose_hiive as mlrose
import numpy as np


"""
  ,_     _,
  |\\___//|
  |=6   6=|
  \=._Y_.=/
   )  `  (    ,
  /       \  ((
  |       |   ))
 /| |   | |\_//
 \| |._.| |/-`
  '"'   '"'
"""

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)
schedule = mlrose.ExpDecay()
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness, _ = mlrose.simulated_annealing(
    problem,
    schedule=schedule,
    max_attempts=10,
    max_iters=1000,
    init_state=init_state,
    random_state=1,
)
print("The best state found is: ", best_state)
