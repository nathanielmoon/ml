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
def run_all(ctx):
    from src.experiments import run
    run()
