
import numpy as np
import pandas as pd
import pickle


def main():
    X_train_inputfile = "../dataset/2.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/2.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/2.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/2.1_y_valid.csv.gz"

    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    knn_pkl = '../models/knn_classifier.pkl'
    knn_model = pickle.load(open(knn_pkl, 'rb'))

    print("Validation score on training dataset (KNN):", knn_model.score(X_train, y_train))
    print("Validation score on testing dataset (KNN):", knn_model.score(X_valid, y_valid))

if __name__ == '__main__':
    main()