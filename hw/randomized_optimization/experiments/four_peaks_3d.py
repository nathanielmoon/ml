import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import seaborn as sns

from paths import OUTPUT


def fitness(leading_zeros, trailing_ones):
    bonus = 100
    b = 0
    T = 10

    if leading_zeros > T and trailing_ones > T:
        b = bonus

    return leading_zeros + trailing_ones + b


def run():
    plt.clf()
    sns.set()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    L = 100
    x = np.arange(0, L / 2, 1)
    y = np.arange(0, L / 2, 1)
    X, Y = np.meshgrid(x, y)

    """
    Z = []
    for i in range(0, len(X)):
        Z.append(fitness(x[i], y[i]))
    Z = np.array(Z)
    """
    Z = X + Y

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(OUTPUT / "fourpeaks-3d.png")
