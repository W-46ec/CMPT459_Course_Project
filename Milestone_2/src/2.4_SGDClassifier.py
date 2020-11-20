
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier


def main():
    X_train_inputfile = "../dataset/2.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/2.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/2.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/2.1_y_valid.csv.gz"

    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    # classify with Stochastic Gradient Descent model
    sgd_model = make_pipeline(
        SGDClassifier()
    )

    sgd_model.fit(X_train, y_train)

    print("Validation score (SGD, train):", sgd_model.score(X_train, y_train))
    print("Validation score (SGD, test):", sgd_model.score(X_valid, y_valid))
    
    
if __name__ == '__main__':
    main()