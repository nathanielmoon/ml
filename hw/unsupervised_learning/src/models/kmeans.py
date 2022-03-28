import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score
import numpy as np
from simple_chalk import magenta

from src.util import resplit_data


def train_with_params(X, params):
    kmeans = KMeans(**params)
    start = time.time_ns()
    kmeans.fit(X)
    end = time.time_ns()
    elapsed_ms = (end - start) / 1000000

    cluster_labels = kmeans.predict(X)
    ss = silhouette_score(X, cluster_labels)

    return kmeans, ss, elapsed_ms


def evaluate(model, X, y):
    y_ = model.predict(X)
    rs = rand_score(y, y_)
    print("\t\tRand Score:", rs)
    return rs


def run_training(data, seed=42, plot=True, output_prefix=""):
    print("\tRunning", magenta("K-Means"))
    np.random.seed(seed)

    X, y, _ = data
    ks = list(range(2, 11))
    results = []
    for k in ks:
        model, sil_score, train_time = train_with_params(X, {"n_clusters": k})
        results.append((k, model, sil_score))
        print(
            "\t\tK =",
            k,
            "| Silhouette Score =",
            sil_score,
            "| Train Time (ms) =",
            train_time,
        )

    results.sort(key=lambda item: item[2], reverse=True)
    best = results[0]
    print(f"\t\tBest: K =", best[0], " | SS =", best[2])

    rs = evaluate(best[1], X, y)

    return {"results": results, "best": best, "randscore": rs}
