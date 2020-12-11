import numpy as np
import pandas as pd
import scipy.stats as stats
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def main():
    X_train_inputfile = "../dataset/3.1_X_train.csv.gz"
    X_valid_inputfile = "../dataset/3.1_X_valid.csv.gz"
    y_train_inputfile = "../dataset/3.1_y_train.csv.gz"
    y_valid_inputfile = "../dataset/3.1_y_valid.csv.gz"
    X_train = pd.read_csv(X_train_inputfile)
    X_valid = pd.read_csv(X_valid_inputfile)
    y_train = pd.read_csv(y_train_inputfile).transpose().values[0]
    y_valid = pd.read_csv(y_valid_inputfile).transpose().values[0]

    knn_model = KNeighborsClassifier(algorithm = 'auto')
    param_dist = {'n_neighbors': stats.uniform(5,200),
                  'weights': ['uniform', 'distance'],
                  'leaf_size': stats.uniform(10,100)}
    n_iter_search = 10
    random_search = RandomizedSearchCV(knn_model, param_distributions = param_dist, n_iter = n_iter_search)
    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    
    '''
    knn_model.fit(X_train, y_train)
    prediction_train = knn_model.predict(X_train)
    prediction_valid = knn_model.predict(X_valid)
    '''
if __name__ == '__main__':
    main()