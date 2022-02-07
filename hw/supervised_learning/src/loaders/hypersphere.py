
import math

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.model_selection import train_test_split


from src.util import upsert_directory
from src.paths import OUTPUT_DIR


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


def find_hypersphere_radius(n, e, r):
    hypersphere_partial = (math.pi ** (n / 2)) / gamma((n / 2) + 1)
    hypercube_volume = volume_of_hypercube(e, n)
    Rn = (hypercube_volume * r) / (hypersphere_partial)
    R = Rn ** (1 / n)
    return R


def generate_dataset(
    n_dimensions=10,
    n_samples=1000,
    boundary_clearance=0.0,
    n_classes=2,
    **kwargs
):
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_dimensions, ))

    origin = np.zeros((n_samples, n_dimensions, ))
    distances = n_dimensional_euc_distance(X, origin)
    thresholds = [
        find_hypersphere_radius(n_dimensions, 2, r=c/n_classes)
        for c in range(1, n_classes)
    ]

    y = np.digitize(distances, thresholds)

    return X, y


def load_data(**kwargs):
    X, y = generate_dataset(**kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def plot_dataset(X, y):
    sns.set()
    sns.set_palette(sns.color_palette("husl", 10))
    upsert_directory(OUTPUT_DIR)
    df = pd.DataFrame(X)
    df = df.rename(columns={0: 'x1', 1: 'x2'})
    df['y'] = y

    sns.scatterplot(data=df, x="x1", y="x2", hue="y",
                    # palette=sns.color_palette(None),
                    s=10)
    plt.title('2D Hypersphere Dataset')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.savefig(OUTPUT_DIR / '2d-hypersphere-dataset.png')
