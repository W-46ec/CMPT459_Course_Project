
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def main():
    X_train_inputfile = "../dataset/2.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/2.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/2.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/2.1_y_valid.csv.gz"

    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    ada_pkl = '../models/ada_classifier.pkl'
    ada_model = pickle.load(open(ada_pkl, 'rb'))

    prediction_train = ada_model.predict(X_train)
    prediction_valid = ada_model.predict(X_valid)

    print("Accuracy score (ADA, train):", round(accuracy_score(y_train, prediction_train), 4))
    print("Accuracy score (ADA, test):", round(accuracy_score(y_train, prediction_train), 4))

    print("Precision score (ADA, train):", round(precision_score(y_train, prediction_train, average = 'weighted'), 4))
    print("Precision score (ADA, test):", round(precision_score(y_train, prediction_train, average = 'weighted'), 4))

    print("Recall score (ADA, train):", round(recall_score(y_train, prediction_train, average = 'weighted'), 4))
    print("Recall score (ADA, test):", round(recall_score(y_valid, prediction_valid, average = 'weighted'), 4))

    print("F score (ADA, train):", round(f1_score(y_train, prediction_train, average = 'weighted'), 4))
    print("F score (ADA, test):", round(f1_score(y_valid, prediction_valid, average = 'weighted'), 4))

if __name__ == '__main__':
    main()