from .dt import run as run_dt
from .bdt import run as run_bdt
from .knn import run as run_knn
from .svm import run as run_svm
from .nn import run as run_nn


def run():
    run_dt()
    run_bdt()
    run_knn()
    run_svm()
    run_nn()
