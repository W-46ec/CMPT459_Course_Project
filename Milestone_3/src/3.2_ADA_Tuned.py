import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def _recall_on_deceased(y, y_pred, **kwargs):
        y_series = pd.Series(y)
        y_deceased = y_series[y_series == 0]
        y_pred_deceased = pd.Series(y_pred)[y_deceased.index]
        return recall_score(
            y_true = y_deceased, 
            y_pred = y_pred_deceased, 
            average = 'micro'
        )

def main():
    X_train_inputfile = "../dataset/3.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/3.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/3.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/3.1_y_valid.csv.gz"
    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    # classify with ADABoost
    ada_model = AdaBoostClassifier(n_estimators=229, algorithm='SAMME', learning_rate=0.850936)
    ada_model.fit(X_train, y_train)

    # predict on test dataset
    pred_valid = ada_model.predict(X_valid)

    print("Accuracy score (ADA):", round(accuracy_score(y_valid, pred_valid),4))
    print("Precision score (ADA):", round(precision_score(y_valid, pred_valid, average = 'weighted'), 4))
    print("Recall score (ADA):", round(recall_score(y_valid, pred_valid, average = 'weighted'), 4))
    print("Recall score on 'deceased' (ADA):", round(_recall_on_deceased(y_valid, pred_valid), 4))

if __name__ == '__main__':
    main()