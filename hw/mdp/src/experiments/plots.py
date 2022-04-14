import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

from src.paths import OUTPUT_DIR

direction_map = {"L": "←", "D": "↓", "R": "→", "U": "↑"}


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


def plot_lotr_policy(matrix, world):
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
            p.set_facecolor(color_map[world[i, j]])
            ax.add_patch(p)

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

    plt.axis("off")
    plt.xlim((0, matrix.shape[1]))
    plt.ylim((0, matrix.shape[0]))
    plt.savefig(OUTPUT_DIR / "lotr-small-policy.png")


def plot_iterative_curve(
    steps,
    data,
    xlabel="",
    ylabel="",
    title="",
    filename="",
):
    plt.clf()
    mean = np.mean(data, axis=1)
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
