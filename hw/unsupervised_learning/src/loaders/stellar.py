import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.paths import DATA_DIR, OUTPUT_DIR

DATA_FILE = DATA_DIR / "stellar/star_classification.csv"

CLASS_FIELD = "class"
IGNORE_FIELDS = ["obj_ID", "run_ID", "rerun_ID", "spec_obj_ID"]


def normalize_data(df):
    for field in df.columns:
        scaler = StandardScaler()
        values = df[field].to_numpy().reshape(-1, 1)
        normed_values = pd.Series(np.squeeze(scaler.fit_transform(values)))
        df[field] = normed_values

    return df


def clean_data(df):
    # TODO maybe experiment with strategies here
    # Try omitting dirty data, since we are likely
    # introducing very bad data
    imputer = SimpleImputer(strategy="most_frequent")
    df.iloc[:, :] = imputer.fit_transform(df)
    return df


def parse_y(df):
    y = df[CLASS_FIELD].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y


def parse_X(df, normalize):
    df = df.drop(columns=[CLASS_FIELD])

    if normalize:
        df = normalize_data(df)

    return df


def load_data(normalize=True):
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=IGNORE_FIELDS)
    df = clean_data(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = parse_y(df)
    X = parse_X(df, normalize)

    return X.to_numpy()[:1000], y[:1000], X.columns
