
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.paths import DATA_DIR, OUTPUT_DIR

DATA_FILE = DATA_DIR / 'penguins/penguins_size.csv'


def normalize_data(df):
    numeric_fields = ['culmen_length_mm', 'culmen_depth_mm',
                      'flipper_length_mm', 'body_mass_g']

    for field in numeric_fields:
        scaler = StandardScaler()
        values = df[field].to_numpy().reshape(-1, 1)
        normed_values = pd.Series(np.squeeze(scaler.fit_transform(values)))
        df[field] = normed_values

    return df


def clean_data(df):
    # TODO maybe experiment with strategies here
    # Try omitting dirty data, since we are likely
    # introducing very bad data
    imputer = SimpleImputer(strategy='most_frequent')
    df.iloc[:, :] = imputer.fit_transform(df)
    return df


def parse_y(df):
    y = df['species'].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y


def parse_X(df):
    df = df.drop(columns=['species'])

    df['island'] = LabelEncoder().fit_transform(df.island.values)
    df['sex'] = LabelEncoder().fit_transform(df.sex.values)

    return df.to_numpy()


def load_data(normalize=False):
    df = pd.read_csv(DATA_FILE)
    df = clean_data(df)

    if normalize:
        df = normalize_data(df)
        print("NROMALZIE")
        print(df)

    y = parse_y(df)
    X = parse_X(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def plot_data():
    sns.set()
    df = pd.read_csv(DATA_FILE)
    df = clean_data(df)

    plt.clf()
    sns.scatterplot(data=df, x="culmen_length_mm",
                    y="culmen_depth_mm", hue="species")
    plt.title('Culmen Length vs Culmen Depth')
    plt.xlabel('Culmen Length')
    plt.ylabel('Culmen Depth')
    plt.savefig(OUTPUT_DIR / 'penguins_culmens.png')

    plt.clf()
    sns.scatterplot(data=df, x="culmen_length_mm",
                    y="flipper_length_mm", hue="species")
    plt.title('Culmen Length vs Flipper Depth')
    plt.xlabel('Culmen Length')
    plt.ylabel('Flipper Length')
    plt.savefig(OUTPUT_DIR / 'penguins_flippers.png')
