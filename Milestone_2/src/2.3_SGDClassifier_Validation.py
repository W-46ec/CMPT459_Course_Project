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

    sgd_pkl = '../models/sgd_classifier.pkl'
    sgd_model = pickle.load(open(sgd_pkl, 'rb'))

    prediction_train = sgd_model.predict(X_train)
    prediction_valid = sgd_model.predict(X_valid)

    print("Accuracy score (SGD, train):", round(accuracy_score(y_train, prediction_train), 4))
    print("Accuracy score (SGD, test):", round(accuracy_score(y_train, prediction_train), 4))

    print("Precision score (SGD, train):", round(precision_score(y_train, prediction_train, average = 'weighted'), 4))
    print("Precision score (SGD, test):", round(precision_score(y_train, prediction_train, average = 'weighted'), 4))

    print("Recall score (SGD, train):", round(recall_score(y_train, prediction_train, average = 'weighted'), 4))
    print("Recall score (SGD, test):", round(recall_score(y_valid, prediction_valid, average = 'weighted'), 4))

    print("F score (SGD, train):", round(f1_score(y_train, prediction_train, average = 'weighted'), 4))
    print("F score (SGD, test):", round(f1_score(y_valid, prediction_valid, average = 'weighted'), 4))

    
if __name__ == '__main__':
    main()