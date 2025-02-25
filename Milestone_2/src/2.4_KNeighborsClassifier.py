import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def main():
    input_data_file = "../dataset/2.0_cases_cleaned.csv.gz"
    data = pd.read_csv(input_data_file)
    to_encode = ['sex', 'province', 'country', 'outcome']
    le = LabelEncoder()
    for i in range(len(to_encode)):
        data[to_encode[i]] = le.fit_transform(data[to_encode[i]].astype(str))    
    target = pd.Series(data['outcome'].to_numpy())    
    
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
        KNeighborsClassifier(n_neighbors=1000, algorithm = 'ball_tree', leaf_size = 100)
    )
    knn_model.fit(X_train, y_train)
    
    scores = cross_val_score(knn_model, data, target, cv=3)
    print("Cross validation avg score (KNN): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print("Validation score (KNN, train):", knn_model.score(X_train, y_train))
    print("Validation score (KNN, test):", knn_model.score(X_valid, y_valid))


if __name__ == '__main__':
    main()