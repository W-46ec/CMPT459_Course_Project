import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def main():
    X_train_inputfile = "../dataset/3.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/3.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/3.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/3.1_y_valid.csv.gz"
    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    # classify with K Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=17, weights='distance', leaf_size=392)
    knn_model.fit(X_train, y_train)

    # predict on test dataset
    pred_valid = knn_model.predict(X_valid)

    print("Accuracy score (KNN):", round(accuracy_score(y_valid, pred_valid), 4))
    print("Precision score (KNN):", round(precision_score(y_valid, pred_valid, average = 'weighted'), 4))
    print("Recall score (KNN):", round(recall_score(y_valid, pred_valid, average = 'weighted'), 4))

if __name__ == '__main__':
    main()