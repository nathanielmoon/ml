from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loaders.penguins import load_data
from src.analyze_model import analyze_classification
from src.util import upsert_directory
from src.paths import OUTPUT_DIR


def penguin():
  optimal_performance_x = ['DT', 'Boosted DT', 'KNN', 'SVM', 'NN']
  optimal_performance_y = [.965, .982, 1.0, 1.0, 1.0]

  plt.clf()
  plt.bar(optimal_performance_x, optimal_performance_y)
  plt.title('Optimal Test Accuracy by Model for Penguin Dataset')
  plt.xlabel('Model')
  plt.ylabel('Test Accuracy')
  plt.savefig(OUTPUT_DIR / 'penguin_performance_comparison.png')


def sphere():
  optimal_performance_x = ['DT', 'Boosted DT', 'KNN', 'SVM', 'NN']
  optimal_performance_y = [.82, .885, .88, .96, 1.0]

  plt.clf()
  plt.bar(optimal_performance_x, optimal_performance_y)
  plt.title('Optimal Test Accuracy by Model for Hypersphere Dataset')
  plt.xlabel('Model')
  plt.ylabel('Test Accuracy')
  plt.savefig(OUTPUT_DIR / 'sphere_performance_comparison.png')

def run():
  sns.set()
  penguin()
  sphere()