import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import randint
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
#from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def report(results, n_top = 5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_Recall_on_deceased'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Accuracy: {0:.3f}".format(results['mean_test_Accuracy'][candidate]))
            print("Overall recall: {0:.3f}".format(results['mean_test_Recall'][candidate]))
            print("Recall on 'deceased': {0:.3f}".format(results['mean_test_Recall_on_deceased'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def _recall_on_deceased(y, y_pred, **kwargs):
    y_series = pd.Series(y)
    y_deceased = y_series[y_series == 0]
    y_pred_deceased = pd.Series(y_pred)[y_deceased.index]
    return recall_score(
        y_true = y_deceased, 
        y_pred = y_pred_deceased, 
        average = 'micro'
    )

scoring = {
    'Accuracy': make_scorer(accuracy_score), 
    'Recall': make_scorer(
        lambda y, y_pred, **kwargs:
            recall_score(
                y_true = y, 
                y_pred = y_pred, 
                average = 'micro'
            )
    ), 
    'Recall_on_deceased': make_scorer(
        lambda y, y_pred, **kwargs:
            _recall_on_deceased(y, y_pred, **kwargs)
    )
}

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
    param_dist = {
        'n_neighbors': randint(5, 100),
        'leaf_size': randint(10, 500)
    }
    n_iter_search = 50
    
    scoring = {
        'Accuracy': make_scorer(accuracy_score), 
        'Recall': make_scorer(
            lambda y, y_pred, **kwargs:
                recall_score(
                    y_true = y, 
                    y_pred = y_pred, 
                    average = 'micro'
                )
        ), 
        'Recall_on_deceased': make_scorer(
            lambda y, y_pred, **kwargs:
                _recall_on_deceased(y, y_pred, **kwargs)
        )
    }

    random_search = RandomizedSearchCV(
        knn_model, 
        param_distributions = param_dist, 
        n_iter = n_iter_search,
        scoring = scoring, 
        refit = 'Recall_on_deceased'
    )
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