import sys
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats

def main():
    txtfile = '../dataset/1.5_joined_individual_cases_Sep20th2020.csv.gz'
    data = pd.read_csv(txtfile)
    
    # decide to keep 'outcome' column or not below. ( remove .drop() )
    X = data.drop(columns=['outcome'])
    y = data['outcome'].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
    
    '''
    print(X_train)
    print(X_valid)    
    print(y_train)
    print(y_valid)
    '''
if __name__ == '__main__':
    main()    