import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
import pandas as pd

from src.paths import OUTPUT_DIR

direction_map = {2: "←", 1: "↓", 3: "→", 0: "↑"}
forest_action_map = {0: "W", 1: "C"}


def plot_lotr_world(matrix):
    plt.clf()
    set_title = "Small LOTR Map"
    color_map = {
        0: "lightgrey",
        1: "red",
        2: "forestgreen",
        3: "gold",
    }

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, xlim=(0, matrix.shape[1]), ylim=(0, matrix.shape[0]))
    font_size = "x-large"
    if matrix.shape[1] > 16:
        font_size = "small"
    plt.title(set_title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            y = matrix.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, edgecolor="black")
            p.set_facecolor(color_map[matrix[i, j]])
            ax.add_patch(p)

            """
            text = ax.text(
                x + 0.5,
                y + 0.5,
                direction_map[matrix[i, j]],
                weight="bold",
                size=font_size,
                horizontalalignment="center",
                verticalalignment="center",
                color="w",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ]
            )
            """

    plt.axis("off")
    plt.xlim((0, matrix.shape[1]))
    plt.ylim((0, matrix.shape[0]))
    plt.savefig(OUTPUT_DIR / "lotr_world.png")


def plot_lotr_policy(matrix, utility, world, filename, title=""):
    plt.clf()
    color_map = {
        0: "lightgrey",
        1: "red",
        2: "forestgreen",
        3: "gold",
    }

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, xlim=(0, matrix.shape[1]), ylim=(0, matrix.shape[0]))
    font_size = "x-large"
    if matrix.shape[1] > 16:
        font_size = "small"
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            y = matrix.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, edgecolor="black")
            p.set_facecolor(color_map[world[i, j]])
            ax.add_patch(p)

            text = ax.text(
                x + 0.5,
                y + 0.5,
                direction_map[matrix[i, j]] + f"\n{utility[i, j]}",
                weight="bold",
                size=font_size,
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
            )

    plt.axis("off")
    plt.xlim((0, matrix.shape[1]))
    plt.ylim((0, matrix.shape[0]))
    plt.savefig(OUTPUT_DIR / filename)


def plot_forest_policy(matrix, utility, R, filename, title=""):
    plt.clf()
    color_map = {
        0: "forestgreen",
        1: "red",
    }

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, xlim=(0, matrix.shape[0]), ylim=(0, 1))
    font_size = "x-large"
    if matrix.shape[0] > 16:
        font_size = "small"
    for i in range(len(matrix)):
        x = i
        y = 0
        p = plt.Rectangle([x, y], 1, 1, edgecolor="black")
        ax.add_patch(p)
        a = matrix[i]
        text = ax.text(
            x + 0.5,
            y + 0.5,
            f"a = {forest_action_map[a]}\nt = {i}\nr = {R[i][a]}\nu = {utility[i]}",
            weight="bold",
            size=font_size,
            horizontalalignment="center",
            verticalalignment="center",
            color="w",
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ]
        )

    plt.title(title)
    plt.axis("off")
    plt.xlim((0, matrix.shape[0]))
    plt.ylim((0, 1))
    plt.savefig(OUTPUT_DIR / filename)


def encode(value):
    if value > 0:
        return 1
    if value < -50:
        return -1
    return 0


def plot_iterative_curve(
    steps,
    data,
    xlabel="",
    ylabel="",
    title="",
    filename="",
):
    plt.clf()
    mean = np.median(data, axis=1)
    std = np.std(data, axis=1)
    _, axes = plt.subplots(1, 1)
    axes.grid(visible=True)
    axes.fill_between(
        steps,
        mean - std,
        mean + std,
        alpha=0.1,
        color="r",
    )
    axes.plot(steps, mean, "o-", color="r", label=ylabel)
    axes.legend(loc="best")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    plt.title(title)

    plt.savefig(OUTPUT_DIR / filename)


def plot_utilityreward_by_discount(
    utilities, rewards, discounts, title="", filename="", xlabel="Discount"
):
    plt.clf()
    data = pd.DataFrame(
        {
            "Utility": utilities,
            "Reward": rewards,
            xlabel: discounts,
        },
    )
    ax = data.plot(x=xlabel, y="Utility", legend=False, color="g")
    ax2 = ax.twinx()
    data.plot(x=xlabel, y="Reward", ax=ax2, legend=False, color="b")
    ax.set_ylabel("Utility")
    ax2.set_ylabel("Reward")
    ax.figure.legend(loc="upper left")
    plt.title(title)
    plt.savefig(OUTPUT_DIR / filename)


def plot_iters_by_discount(discounts, iters, title="", filename="", xlabel="Discount"):
    plt.clf()
    df = pd.DataFrame({"Iteration": iters, xlabel: discounts})
    sns.lineplot(data=df, x=xlabel, y="Iteration")
    plt.title(title)
    plt.savefig(OUTPUT_DIR / filename)


def plot_time_by_discount(discounts, times, title="", filename=""):
    plt.clf()
    df = pd.DataFrame({"Time (ns)": times, "Discount": discounts})
    sns.lineplot(data=df, x="Discount", y="Time (ns)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)


def plot_delta_by_iterations(deltas, title="", filename=""):
    plt.clf()
    df = pd.DataFrame({"Delta": deltas, "Iteration": [x for x in range(len(deltas))]})
    sns.lineplot(data=df, x="Iteration", y="Delta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)


def plot_utilityreward_by_iterations(utilities, rewards, title="", filename=""):
    plt.clf()
    data = pd.DataFrame(
        {
            "Utility": utilities,
            "Reward": rewards,
            "Iteration": [x for x in range(len(rewards))],
        },
    )
    ax = data.plot(x="Iteration", y="Utility", legend=False, color="g")
    ax2 = ax.twinx()
    data.plot(x="Iteration", y="Reward", ax=ax2, legend=False, color="b")
    ax.set_ylabel("Utility")
    ax2.set_ylabel("Reward")
    ax.figure.legend(loc="upper left")
    plt.tight_layout()
    plt.title(title)
    plt.savefig(OUTPUT_DIR / filename)
