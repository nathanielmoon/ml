from simple_chalk import cyan, magenta, white
import seaborn as sns
import pandas as pd
import numpy as np

from src.paths import OUTPUT_DIR
from src.util import upsert_directory
from src.loaders.penguins import load_data as load_penguin_data
from src.loaders.stellar import load_data as load_stellar_data
from src.models.kmeans import train_with_params as train_kmeans
from src.models.em import train_with_params as train_em
from src.models.pca import train_with_params as train_pca
from src.models.ica import train_with_params as train_ica
from src.models.rp import train_with_params as train_rp
from src.models.dt import train_with_params as train_dt
from src.models.nn import train_with_params as train_nn
from src.experiments.plots import ex3_plot_learning_curve
from src.terms import (
    PENGUINS_DS_LABEL,
    STELLAR_DS_LABEL,
    KMEANS_LABEL,
    EM_LABEL,
    PCA_LABEL,
    ICA_LABEL,
    RP_LABEL,
    IME_LABEL,
    DT_LABEL,
)


def to_snake_case(s):
    return s.replace(" ", "_").lower()


def run():
    sns.set()
    upsert_directory(OUTPUT_DIR)

    penguin_data = load_penguin_data()
    X, y, feature_labels = penguin_data

    cluster_algorithms = [
        (KMEANS_LABEL, train_kmeans, {"n_clusters": 2}),
        (EM_LABEL, train_em, {"n_components": 10}),
    ]
    dim_red_algorithms = [
        (
            PCA_LABEL,
            train_pca,
            {"n_components": 3},
        ),
        (
            ICA_LABEL,
            train_ica,
            {"n_components": 1},
        ),
        (RP_LABEL, train_rp, {"n_components": 5}),
        (
            DT_LABEL,
            train_dt,
            {"features": ["flipper_length_mm", "culmen_length_mm", "island"]},
        ),
    ]

    results = {}
    bests = []

    results["Original"] = train_nn(penguin_data, {})
    train_sizes, train_scores, test_scores, fit_times = results["Original"]

    bests.append(
        {
            "Dataset": "Original",
            "Train Score": np.mean(train_scores[-1]),
            "Test Score": np.mean(test_scores[-1]),
        }
    )

    for dr_label, dr_algo, dr_params in dim_red_algorithms:
        print(
            "Running with dataset:",
            magenta(dr_label),
        )
        dr_model = dr_algo(penguin_data, dr_params)[0]
        X_ = dr_model.transform(X)
        results[dr_label] = train_nn((X_, y, feature_labels), {})
        train_sizes, train_scores, test_scores, fit_times = results[dr_label]
        bests.append(
            {
                "Dataset": dr_label,
                "Train Score": np.mean(train_scores[-1]),
                "Test Score": np.mean(test_scores[-1]),
            }
        )

    for c_label, c_algo, c_params in cluster_algorithms:
        print(
            "Running with dataset:",
            magenta(c_label),
        )
        cluster_model = c_algo(X, c_params)[0]
        y_ = cluster_model.predict(X)
        X_ = np.append(X, np.expand_dims(y_, axis=1), 1)
        print("LOOKE HERE", X.shape, X_.shape)
        results[c_label] = train_nn((X_, y, feature_labels), {})
        train_sizes, train_scores, test_scores, fit_times = results[c_label]
        bests.append(
            {
                "Dataset": c_label,
                "Train Score": np.mean(train_scores[-1]),
                "Test Score": np.mean(test_scores[-1]),
            }
        )

    best_df = pd.DataFrame(bests)
    best_df.to_csv(OUTPUT_DIR / "ex3_bests.csv")
    print(best_df)
    plots(results, penguin_data)


def plots(results, penguin_data):
    print("Plotting all the things ...")

    for k, v in results.items():
        outstring = to_snake_case(k)
        title = f"Penguin {k} Data Learning Curve"
        ex3_plot_learning_curve(
            *v, out=f"ex3_{outstring}_learning_curve.png", title=title
        )
