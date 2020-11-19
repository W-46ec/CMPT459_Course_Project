import sys
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats

def main():
    txtfile = "../dataset/2.0_cases_cleaned.csv.gz"
    data = pd.read_csv(txtfile)

    to_encode = ['sex', 'province', 'country', 'outcome']
    le = LabelEncoder()
    for i in range(len(to_encode)):
        data[to_encode[i]] = le.fit_transform(data[to_encode[i]].astype(str))
    
    X = data
    y = data['outcome'].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
    
    # classify with Gaussian Naive Bayes model
    nb_model = make_pipeline(
        GaussianNB()
        )  
    
    # classify with KNN model
    knn_model = make_pipeline(
        KNeighborsClassifier()
        )
    
    # classify with ADABoost
    ada_model = make_pipeline(
        AdaBoostClassifier()
        )
    
    nb_model.fit(X_train, y_train)
    print("Validation score (NB):", nb_model.score(X_valid, y_valid))  
    
    knn_model.fit(X_train, y_train)
    print("Validation score (KNN):", knn_model.score(X_valid, y_valid)) 
    
    ada_model.fit(X_train, y_train)
    print("Validation score (ADA):", ada_model.score(X_valid, y_valid))  
    
    nb_pkl = '../models/nb_classifier.pkl'
    pickle.dump(nb_model, open(nb_pkl, 'wb'))
    #knn_pkl = '../models/knn_classifier.pkl'
    #pickle.dump(knn_model, open(knn_pkl, 'wb'))
    ada_pkl = '../models/ada_classifier.pkl'
    pickle.dump(ada_model, open(ada_pkl, 'wb'))
    

if __name__ == '__main__':
    main()    