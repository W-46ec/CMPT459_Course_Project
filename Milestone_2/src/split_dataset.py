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

    '''
    le = LabelEncoder()
    le.fit(data['sex'].astype(str))
    data['sex'] = le.transform(data['sex'].astype(str))
    print(data['sex'])
    
    test = pd.get_dummies(data['sex'])
    print(test)
    '''
    X = data
    y = data['outcome'].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
    
    # classify with Gaussian Naive Bayes model
    nb_model = make_pipeline(
        GaussianNB()
        )  
    
    # classify with KNN model
    knn_model = make_pipeline(
        KNeighborsClassifier(n_neighbors=20)
        )  
    
    #nb_model.fit(X_train, y_train)
    #print("Validation score:", nb_model.score(X_valid, y_valid))  
    
if __name__ == '__main__':
    main()    