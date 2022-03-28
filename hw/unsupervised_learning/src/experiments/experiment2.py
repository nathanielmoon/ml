from simple_chalk import cyan, magenta, white
import seaborn as sns
import pandas as pd

from src.paths import OUTPUT_DIR
from src.util import upsert_directory
from src.loaders.penguins import load_data as load_penguin_data
from src.loaders.stellar import load_data as load_stellar_data
from src.models.kmeans import run_training as train_kmeans
from src.models.em import run_training as train_em
from src.models.pca import train_with_params as train_pca
from src.models.ica import train_with_params as train_ica
from src.models.rp import train_with_params as train_rp
from src.models.dt import train_with_params as train_dt
from src.experiments.plots import (
    ex1_pairplots,
    ex1_plot_kmeans_sil_score,
    ex1_plot_em_ll_score,
    ex1_plot_kmeans_clusters,
    ex1_plot_em_clusters,
    ex1_plot_pca_variances,
    ex1_plt_ica_stats,
    ex1_plot_rp_error,
    ex1_plot_dt_importances,
)
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


def run():
    sns.set()
    upsert_directory(OUTPUT_DIR)

    penguin_data = load_penguin_data()
    stellar_data = load_stellar_data()

    datasets = [(PENGUINS_DS_LABEL, penguin_data), (STELLAR_DS_LABEL, stellar_data)]
    cluster_algorithms = [
        (KMEANS_LABEL, train_kmeans),
        (EM_LABEL, train_em),
    ]
    dim_red_algorithms = [
        (
            PCA_LABEL,
            train_pca,
            {
                PENGUINS_DS_LABEL: {"n_components": 3},
                STELLAR_DS_LABEL: {"n_components": 7},
            },
        ),
        (
            ICA_LABEL,
            train_ica,
            {
                PENGUINS_DS_LABEL: {"n_components": 1},
                STELLAR_DS_LABEL: {"n_components": 12},
            },
        ),
        (
            RP_LABEL,
            train_rp,
            {
                PENGUINS_DS_LABEL: {"n_components": 5},
                STELLAR_DS_LABEL: {"n_components": 8},
            },
        ),
        (
            DT_LABEL,
            train_dt,
            {
                PENGUINS_DS_LABEL: {
                    "features": ["flipper_length_mm", "culmen_length_mm", "island"]
                },
                STELLAR_DS_LABEL: {"features": ["redshift", "z", "g"]},
            },
        ),
    ]

    results = {
        x: {y: {y: None for y, _ in datasets} for y, _, _ in dim_red_algorithms}
        for x, _ in cluster_algorithms
    }

    rand_scores = []
    for c_label, c_algorithm in cluster_algorithms:
        for dr_label, dr_algo, params in dim_red_algorithms:
            for ds_label, data in datasets:
                dr_params = params[ds_label]
                print(
                    "Running with dataset:",
                    white(c_label),
                    magenta(dr_label),
                    cyan(ds_label),
                )

                X, y, features = data
                dr_model = dr_algo(data, dr_params)[0]
                X_ = dr_model.transform(X)
                result = c_algorithm((X_, y, features))
                results[c_label][dr_label][ds_label] = results
                rand_scores.append(
                    {
                        "Clusterer": c_label,
                        "Dimensionality Reducer": dr_label,
                        "Dataset": ds_label,
                        "K": result["best"][0],
                        "Rand Index": result["randscore"],
                    },
                )

    rand_scores = pd.DataFrame(rand_scores)
    rand_scores.to_csv(OUTPUT_DIR / "ex1_randscores.csv")
    print("RAND SCORES")
    print(rand_scores)
    plots(results, penguin_data, stellar_data)


def plots(results, penguin_data, stellar_data):
    print("Plotting all the things ...")
