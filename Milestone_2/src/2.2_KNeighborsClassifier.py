
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier


def main():
    X_train_inputfile = "../dataset/2.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/2.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/2.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/2.1_y_valid.csv.gz"

    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    # classify with KNN model
    knn_model = make_pipeline(
        KNeighborsClassifier()
    )

    knn_model.fit(X_train, y_train)

    knn_pkl = '../models/knn_classifier.pkl'
    pickle.dump(knn_model, open(knn_pkl, 'wb'))

if __name__ == '__main__':
    main()