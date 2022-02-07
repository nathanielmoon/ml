from invoke import task
import numpy as np


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


@task
def run(ctx, name):
    run = import_from(name, "run")
    run()


@task
def boop(ctx):
    from src.loaders.hypersphere import load_data, plot_dataset
    load_data()
    np.random.seed(42)
    X, _, y, _ = load_data(
        n_dimensions=2,
        n_samples=10000,
        n_classes=4
    )
    plot_dataset(X, y)


@task
def plot_penguins(ctx):
    from src.loaders.penguins import plot_data
    plot_data()
