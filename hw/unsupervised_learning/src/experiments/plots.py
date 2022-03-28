import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance

from src.paths import OUTPUT_DIR
from src.terms import (
    PENGUINS_DS_LABEL,
    STELLAR_DS_LABEL,
    KMEANS_LABEL,
    EM_LABEL,
    PCA_LABEL,
    ICA_LABEL,
    RP_LABEL,
    DT_LABEL,
)


def ex1_pairplots(results, penguin_data, stellar_data):
    # Normal Pairplot Penguin
    plt.clf()
    df = pd.DataFrame(penguin_data[0], columns=penguin_data[2])
    df["species"] = penguin_data[1]
    df = df.drop(columns=["sex", "island"])
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_pairplot.png")

    # KMeans Pairplot Penguin
    plt.clf()
    X = penguin_data[0]
    y = results[PENGUINS_DS_LABEL][KMEANS_LABEL]["best"][1].predict(X)
    df = pd.DataFrame(penguin_data[0], columns=penguin_data[2])
    df["species"] = y
    df = df.drop(columns=["sex", "island"])
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_kmeans_pairplot.png")

    # EM Pairplot Penguin
    plt.clf()
    X = penguin_data[0]
    y = results[PENGUINS_DS_LABEL][EM_LABEL]["best"][1].predict(X)
    df = pd.DataFrame(penguin_data[0], columns=penguin_data[2])
    df["species"] = y
    df = df.drop(columns=["sex", "island"])
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_em_pairplot.png")

    plt.clf()
    df = pd.DataFrame(stellar_data[0], columns=stellar_data[2])
    df["class"] = stellar_data[1]
    df = df.drop(columns=["cam_col", "fiber_ID", "plate", "field_ID", "MJD"])
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_pairplot.png")

    # KMeans Pairplot Penguin
    plt.clf()
    X = stellar_data[0]
    y = results[STELLAR_DS_LABEL][KMEANS_LABEL]["best"][1].predict(X)
    df = pd.DataFrame(stellar_data[0], columns=stellar_data[2])
    df = df.drop(columns=["cam_col", "fiber_ID", "plate", "field_ID", "MJD"])
    df["class"] = y
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_kmeans_pairplot.png")

    # EM Pairplot Penguin
    plt.clf()
    X = stellar_data[0]
    y = results[STELLAR_DS_LABEL][EM_LABEL]["best"][1].predict(X)
    df = pd.DataFrame(stellar_data[0], columns=stellar_data[2])
    df = df.drop(columns=["cam_col", "fiber_ID", "plate", "field_ID", "MJD"])
    df["class"] = y
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_em_pairplot.png")


def ex1_plot_kmeans_sil_score(results):
    plt.clf()
    df = pd.DataFrame(
        {
            PENGUINS_DS_LABEL: [
                x[2] for x in results[PENGUINS_DS_LABEL][KMEANS_LABEL]["results"]
            ],
            STELLAR_DS_LABEL: [
                x[2] for x in results[STELLAR_DS_LABEL][KMEANS_LABEL]["results"]
            ],
            "K": [x[0] for x in results[PENGUINS_DS_LABEL][KMEANS_LABEL]["results"]],
        }
    )
    df = df.set_index(["K"])

    sns.lineplot(data=df)
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Score by K")
    plt.savefig(OUTPUT_DIR / "ex1_kmeans_silscore.png")


def ex1_plot_em_ll_score(results):
    plt.clf()
    df = pd.DataFrame(
        {
            PENGUINS_DS_LABEL: [
                x[2] for x in results[PENGUINS_DS_LABEL][EM_LABEL]["results"]
            ],
            STELLAR_DS_LABEL: [
                x[2] for x in results[STELLAR_DS_LABEL][EM_LABEL]["results"]
            ],
            "K": [x[0] for x in results[PENGUINS_DS_LABEL][EM_LABEL]["results"]],
        }
    )
    df = df.set_index(["K"])

    sns.lineplot(data=df)
    plt.xlabel("K")
    plt.ylabel("Log Likelihood")
    plt.title("Expectation Maximization Log Likelihood by K")
    plt.savefig(OUTPUT_DIR / "ex1_em_loglike.png")

    plt.clf()
    df = pd.DataFrame(
        {
            PENGUINS_DS_LABEL: [
                x[3] for x in results[PENGUINS_DS_LABEL][EM_LABEL]["results"]
            ],
            STELLAR_DS_LABEL: [
                x[3] for x in results[STELLAR_DS_LABEL][EM_LABEL]["results"]
            ],
            "K": [x[0] for x in results[PENGUINS_DS_LABEL][EM_LABEL]["results"]],
        }
    )
    df = df.set_index(["K"])

    sns.lineplot(data=df)
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.title("Expectation Maximization BIC by K")
    plt.savefig(OUTPUT_DIR / "ex1_em_bic.png")


def ex1_plot_kmeans_clusters(results, penguin_data, stellar_data):
    custom_lines = [
        Line2D([0], [0], color=(153 / 255, 153 / 255, 153 / 255, 0.6), lw=5),
        Line2D([0], [0], color=(227 / 255, 26 / 255, 27 / 255, 0.6), lw=5),
        Line2D([0], [0], color="r", lw=2, linestyle=(0, (5, 1))),
    ]
    plt.clf()
    visualizer = SilhouetteVisualizer(
        results[PENGUINS_DS_LABEL][KMEANS_LABEL]["best"][1]
    )
    visualizer.fit(penguin_data[0])

    plt.ylabel("Sample")
    plt.xlabel("Silhouette Score")
    plt.title("Silhouette Plot of K-Means for Penguin Data with 2 Clusters")
    legend = plt.legend(
        custom_lines,
        ["Cluster 0", "Cluster 1", "Average Silhouette Score"],
        framealpha=0.6,
        frameon=True,
        loc="upper left",
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    plt.savefig(OUTPUT_DIR / "ex1_kmeans_penguin_clusters.png")

    plt.clf()
    visualizer = SilhouetteVisualizer(
        results[STELLAR_DS_LABEL][KMEANS_LABEL]["best"][1]
    )
    visualizer.fit(stellar_data[0])
    plt.ylabel("Sample")
    plt.xlabel("Silhouette Score")
    plt.title("Silhouette Plot of K-Means for Stellar Data with 2 Clusters")
    legend = plt.legend(
        custom_lines,
        ["Cluster 0", "Cluster 1", "Average Silhouette Score"],
        framealpha=0.6,
        frameon=True,
        loc="upper left",
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    plt.savefig(OUTPUT_DIR / "ex1_kmeans_stellar_clusters.png")


def ex1_plot_em_clusters(results, penguin_data, stellar_data):
    pass


def ex1_plot_pca_variances(results):
    variances = results[PENGUINS_DS_LABEL][PCA_LABEL]["variances"]
    plt.clf()
    components = range(1, len(variances) + 1)
    cum_variance = [sum(variances[0 : i + 1]) for i, _ in enumerate(variances)]
    df = pd.DataFrame(
        {
            "Component": components,
            "Individual Variance": variances,
            "Cumulative Variance": cum_variance,
        }
    )
    df = df.set_index(["Component"])
    sns.lineplot(data=df)
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Penguin Data PCA Variance")
    plt.savefig(OUTPUT_DIR / f"ex1_penguin_pca_variance.png")

    variances = results[STELLAR_DS_LABEL][PCA_LABEL]["variances"]
    plt.clf()
    components = range(1, len(variances) + 1)
    cum_variance = [sum(variances[0 : i + 1]) for i, _ in enumerate(variances)]
    df = pd.DataFrame(
        {
            "Component": components,
            "Individual Variance": variances,
            "Cumulative Variance": cum_variance,
        }
    )
    df = df.set_index(["Component"])
    sns.lineplot(data=df)
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Stellar Data PCA Variance")
    plt.savefig(OUTPUT_DIR / f"ex1_stellar_pca_variance.png")


def ex1_plt_ica_stats(results):
    plt.clf()
    p_results = results[PENGUINS_DS_LABEL][ICA_LABEL]["results"]
    df = pd.DataFrame(
        {
            "Component": [x[0] for x in p_results],
            "Average Kurtosis": [x[2] for x in p_results],
            # "Median Kurtosis": [x[3] for x in p_results],
        }
    )
    df = df.set_index("Component")
    sns.lineplot(data=df)
    plt.xlabel("Component")
    plt.ylabel("Kurtosis")
    plt.title("Penguin Data ICA Kurtosis")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_ica_kurtosis.png")

    plt.clf()
    p_results = results[STELLAR_DS_LABEL][ICA_LABEL]["results"]
    df = pd.DataFrame(
        {
            "Component": [x[0] for x in p_results],
            "Average Kurtosis": [x[2] for x in p_results],
            # "Median Kurtosis": [x[3] for x in p_results],
        }
    )
    df = df.set_index("Component")
    sns.lineplot(data=df)
    plt.xlabel("Component")
    plt.ylabel("Kurtosis")
    plt.title("Stellar Data ICA Kurtosis")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_ica_kurtosis.png")


def ex1_plot_rp_error(results):
    plt.clf()
    errors = [x[3] for x in results[PENGUINS_DS_LABEL][RP_LABEL]["results"]]
    error_mean = np.mean(errors, axis=1)
    error_std = np.std(errors, axis=1)
    points = np.array(range(1, len(errors) + 1))

    _, axes = plt.subplots(1, 1)
    axes.grid(visible=True)
    axes.fill_between(
        points,
        error_mean - error_std,
        error_mean + error_std,
        alpha=0.1,
        color="r",
    )
    axes.plot(
        points,
        error_mean,
        "o-",
        color="r",
        label="Training score",
    )
    axes.set_xlabel("Component")
    axes.set_ylabel("Error")

    plt.title("Penguin Data Randomized Projection Reconstruction Error")

    plt.savefig(OUTPUT_DIR / "ex1_penguin_rp_error.png")

    plt.clf()
    errors = [x[3] for x in results[STELLAR_DS_LABEL][RP_LABEL]["results"]]
    error_mean = np.mean(errors, axis=1)
    error_std = np.std(errors, axis=1)
    points = np.array(range(1, len(errors) + 1))

    _, axes = plt.subplots(1, 1)
    axes.grid(visible=True)
    axes.fill_between(
        points,
        error_mean - error_std,
        error_mean + error_std,
        alpha=0.1,
        color="r",
    )
    axes.plot(
        points,
        error_mean,
        "o-",
        color="r",
        label="Training score",
    )
    axes.set_xlabel("Component")
    axes.set_ylabel("Error")

    plt.title("Stellar Data Randomized Projection Reconstruction Error")

    plt.savefig(OUTPUT_DIR / "ex1_stellar_rp_error.png")


def ex1_plot_dt_importances(results):
    plt.clf()
    importances = results[PENGUINS_DS_LABEL][DT_LABEL]["importances"]
    reductions = [x[1] for x in importances]
    cum_reductions = [sum(reductions[0 : i + 1]) for i, _ in enumerate(reductions)]
    labels = [x[0] for x in importances]
    print(list(zip(labels, cum_reductions)))
    df = pd.DataFrame(
        {
            "Individual Importance": reductions,
            "Cumulative Importance": cum_reductions,
        },
        index=labels,
    )
    sns.lineplot(data=df)
    plt.xlabel("Feature")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Importance")
    plt.title("Penguin Data Decision Tree Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ex1_penguin_dt_importance.png")

    plt.clf()
    importances = results[STELLAR_DS_LABEL][DT_LABEL]["importances"]
    reductions = [x[1] for x in importances]
    cum_reductions = [sum(reductions[0 : i + 1]) for i, _ in enumerate(reductions)]
    labels = [x[0] for x in importances]
    print(list(zip(labels, cum_reductions)))
    df = pd.DataFrame(
        {
            "Individual Importance": reductions,
            "Cumulative Importance": cum_reductions,
        },
        index=labels,
    )
    sns.lineplot(data=df)
    plt.xlabel("Feature")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Importance")
    plt.title("Stellar Data Decision Tree Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ex1_stellar_dt_importance.png")


def ex3_plot_learning_curve(
    train_sizes,
    train_scores,
    test_scores,
    fit_times,
    out="learning_curve",
    title="Learning Curve",
):
    plt.clf()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    _, axes = plt.subplots(1, 1)
    axes.grid(visible=True)
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")
    axes.set_xlabel("Training Examples")
    axes.set_ylabel("F1 Score")

    plt.title(title)

    plt.savefig(OUTPUT_DIR / out)


def ex1_plot_pca_pairwise(results, penguin_data, stellar_data):
    plt.clf()
    X, y, _ = penguin_data
    model = results[PENGUINS_DS_LABEL][PCA_LABEL]["model"]
    X_ = model.transform(X)

    df = pd.DataFrame(X_[:, :3])
    df["species"] = y
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_pca_pairwise.png")

    plt.clf()
    X, y, _ = stellar_data
    model = results[STELLAR_DS_LABEL][PCA_LABEL]["model"]
    X_ = model.transform(X)

    df = pd.DataFrame(X_[:, :7])
    df["class"] = y
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_pca_pairwise.png")


def ex1_plot_ica_pairwise(results, penguin_data, stellar_data):
    plt.clf()
    X, y, _ = penguin_data
    model = results[PENGUINS_DS_LABEL][ICA_LABEL]["results"][0][1]
    X_ = model.transform(X)

    df = pd.DataFrame(X_)
    df["species"] = y
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_ica_pairwise.png")

    plt.clf()
    X, y, _ = stellar_data
    model = results[STELLAR_DS_LABEL][ICA_LABEL]["results"][11][1]
    X_ = model.transform(X)

    df = pd.DataFrame(X_)
    df["class"] = y
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_ica_pairwise.png")


def ex1_plot_rp_pairwise(results, penguin_data, stellar_data):
    plt.clf()
    X, y, _ = penguin_data
    model = results[PENGUINS_DS_LABEL][RP_LABEL]["results"][5][1]
    X_ = model.transform(X)

    df = pd.DataFrame(X_)
    df["species"] = y
    sns.pairplot(df, hue="species")
    plt.savefig(OUTPUT_DIR / "ex1_penguin_rp_pairwise.png")

    plt.clf()
    X, y, _ = stellar_data
    model = results[STELLAR_DS_LABEL][RP_LABEL]["results"][8][1]
    X_ = model.transform(X)

    df = pd.DataFrame(X_)
    df["class"] = y
    sns.pairplot(df, hue="class")
    plt.savefig(OUTPUT_DIR / "ex1_stellar_rp_pairwise.png")
