from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsertDirectory
from src.paths import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / 'nn/penguins/'


def run():
    print('ehllp')
