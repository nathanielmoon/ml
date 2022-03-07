from invoke import task


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


@task
def run(ctx, name):
    run = import_from(name, "run")
    run()


@task
def experiment1(ctx):
    from experiments.experiment1 import run
    import numpy as np

    np.random.seed(42)

    run()


@task
def experiment2(ctx):
    from experiments.experiment2 import run
    import numpy as np

    np.random.seed(42)

    run()
