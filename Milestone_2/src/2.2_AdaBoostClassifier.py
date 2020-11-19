
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier


def main():
    X_train_inputfile = "../dataset/2.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/2.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/2.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/2.1_y_valid.csv.gz"

    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    # classify with ADABoost
    ada_model = make_pipeline(
        AdaBoostClassifier()
    )

    ada_model.fit(X_train, y_train)

    ada_pkl = '../models/ada_classifier.pkl'
    pickle.dump(ada_model, open(ada_pkl, 'wb'))

if __name__ == '__main__':
    main()