from invoke import task


@task
def hello(ctx):
    from src.loaders.penguins import load_data
    load_data()


@task
def penguins(ctx):
    from src.experiments.penguins import run
    run()
