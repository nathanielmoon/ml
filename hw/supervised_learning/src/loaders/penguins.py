
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.paths import DATA_DIR

DATA_FILE = DATA_DIR / 'penguins/penguins_size.csv'


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


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = clean_data(df)

    y = parse_y(df)
    X = parse_X(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
