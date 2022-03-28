from simple_chalk import cyan
import seaborn as sns

from src.paths import OUTPUT_DIR
from src.util import upsert_directory
from src.loaders.penguins import load_data as load_penguin_data
from src.loaders.stellar import load_data as load_stellar_data
from src.models.kmeans import run_training as train_kmeans
from src.models.em import run_training as train_em
from src.models.pca import run_training as train_pca
from src.models.ica import run_training as train_ica
from src.models.rp import run_training as train_rp
from src.models.dt import run_training as train_dt
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
    ex1_plot_pca_pairwise,
    ex1_plot_ica_pairwise,
    ex1_plot_rp_pairwise,
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
        (PCA_LABEL, train_pca),
        (ICA_LABEL, train_ica),
        (RP_LABEL, train_rp),
        (DT_LABEL, train_dt),
    ]

    results = {
        x: {y: None for y, _ in cluster_algorithms + dim_red_algorithms}
        for x, _ in datasets
    }

    for ds_label, data in datasets:
        print("Running with dataset:", cyan(ds_label))

        for a_label, algorithm in cluster_algorithms + dim_red_algorithms:
            results[ds_label][a_label] = algorithm(
                data, plot=True, output_prefix=f"ex1-{ds_label}-{a_label}"
            )

    plots(results, penguin_data, stellar_data)


def plots(results, penguin_data, stellar_data):
    print("Plotting all the things ...")
    ex1_plot_kmeans_sil_score(results)
    ex1_plot_em_ll_score(results)
    ex1_plot_kmeans_clusters(results, penguin_data, stellar_data)
    ex1_plot_em_clusters(results, penguin_data, stellar_data)
    ex1_plot_pca_variances(results)
    ex1_plt_ica_stats(results)
    ex1_plot_rp_error(results)
    ex1_plot_dt_importances(results)
    ex1_plot_pca_pairwise(results, penguin_data, stellar_data)
    ex1_plot_ica_pairwise(results, penguin_data, stellar_data)
    ex1_plot_rp_pairwise(results, penguin_data, stellar_data)
    ex1_pairplots(results, penguin_data, stellar_data)
