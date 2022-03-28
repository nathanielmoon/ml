import time

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, rand_score
import numpy as np
from simple_chalk import magenta

from src.util import resplit_data


def train_with_params(X, params):
    em = GaussianMixture(**params)
    start = time.time_ns()
    em.fit(X)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    loglikelihood = em.score(X)
    bic = em.bic(X)

    return em, loglikelihood, bic, elapsed_ms


def evaluate(model, X, y):
    y_ = model.predict(X)
    rs = rand_score(y, y_)
    print("\t\tRand Score:", rs)
    return rs


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("Expectation Maximization"))
    np.random.seed(seed)

    X, y, _ = data
    ks = list(range(2, 11))
    results = []
    for k in ks:
        model, ll, bic, train_time = train_with_params(X, {"n_components": k})
        results.append((k, model, ll, bic))
        print(
            "\t\tK =",
            k,
            "| Log Likelihood =",
            ll,
            "| BIC =",
            bic,
            "| Train Time (ms) =",
            train_time,
        )

    results.sort(key=lambda item: item[3])
    best = results[0]
    print(f"\t\tBest: K =", best[0], " | BIC =", best[3])

    rs = evaluate(best[1], X, y)

    return {"results": results, "best": best, "randscore": rs}
