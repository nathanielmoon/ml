from .dt import run as run_dt
from .bdt import run as run_bdt
from .knn import run as run_knn
from .svm import run as run_svm
from .nn import run as run_nn

from .bdt_iterations import run as bdti_run
from .nn_iterations import run as nni_run
from .size_and_time import run as sat_run
from .model_comparison_charts import run as mcc_run

def run():
    run_dt()
    run_bdt()
    run_knn()
    run_svm()
    run_nn()
    bdti_run()
    nni_run()
    sat_run()
    mcc_run()

