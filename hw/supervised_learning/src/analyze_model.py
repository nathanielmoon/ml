from cgi import test
import json

import pandas as pd
from sklearn.metrics import classification_report


def analyze_classification(run, X_train, X_test, y_train, y_test):
    # run fn should be (X, y) => model

    # Train data report
    y_pred = run(X_train)
    train_report = classification_report(y_train, y_pred, output_dict=True)
    train_report = pd.DataFrame(train_report).transpose()
    train_report = train_report.reset_index().rename(
        columns={'index': 'class'})
    train_report.insert(0, 'mode', 'train')

    # Test data report
    y_pred = run(X_test)
    test_report = classification_report(y_test, y_pred, output_dict=True)
    test_report = pd.DataFrame(test_report).transpose()
    test_report = test_report.reset_index().rename(columns={'index': 'class'})
    test_report.insert(0, 'mode', 'test')

    # Combine report
    report = pd.concat([train_report, test_report], ignore_index=True)
    print(report)
