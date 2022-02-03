
import math

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma


def volume_of_hypersphere(radius, dimensions):
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball#The_volume
    numerator = math.pi ** (dimensions / 2)
    denominator = gamma((dimensions / 2) + 1)
    coefficient = radius ** dimensions
    return (numerator / denominator) * coefficient


def volume_of_hypercube(edge_length, dimensions):
    # https://en.wikipedia.org/wiki/Hypercube#Vertex_coordinates
    return edge_length ** dimensions


def n_dimensional_euc_distance(p1, p2):
    # https://iq.opengenus.org/euclidean-distance/
    return np.sqrt(np.sum(np.square(np.subtract(p1, p2)), axis=1))


def find_hypersphere_radius(n, e):
    hypersphere_partial = (math.pi ** (n / 2)) / gamma((n / 2) + 1)
    hypercube_volume = volume_of_hypercube(e, n)
    Rn = (hypercube_volume / 2) / (hypersphere_partial)
    R = Rn ** (1 / n)
    return R


def generate_dataset(
    n_dimensions=10,
    n_samples=1000,
    boundary_clearance=0.0
):
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_dimensions, ))

    origin = np.zeros((n_samples, n_dimensions, ))
    distances = n_dimensional_euc_distance(X, origin)
    print("----- Distances")
    print(distances)
    threshold = find_hypersphere_radius(n_dimensions, 2)
    print("THRESHOLD", threshold)
    y = np.less_equal(distances, np.full(n_samples, threshold)).astype(int)

    return X, y


def load_data(**kwargs):
    X, y = generate_dataset(**kwargs)

    print("Ratios")
    for n in range(1, 11):
        vhs = volume_of_hypersphere(1, n)
        vhc = volume_of_hypercube(2, n)
        print(f"{n}D: ", vhs, "/", vhc, "=", vhs / vhc)

    print("----- X")
    print(X)
    print("----- y")
    print(y)
    print("Class 0:", y.shape[0] - np.count_nonzero(y))
    print("Class 1:", np.count_nonzero(y))
    return X, y


def plot_dataset(X, y):
    df = pd.DataFrame(X)
    df = df.rename(columns={0: 'x1', 1: 'x2'})
    df['y'] = y

    sns.scatterplot(data=df, x="x1", y="x2", hue="y")
    plt.show()


if __name__ == '__main__':
    sns.set()
    np.random.seed(42)
    X, y = load_data(
        n_dimensions=2,
        n_samples=1000,
        n_classes=2
    )
    plot_dataset(X, y)
