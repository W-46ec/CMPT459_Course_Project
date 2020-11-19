
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


def main():
    input_data_file = "../dataset/2.0_cases_cleaned.csv.gz"
    data = pd.read_csv(input_data_file)

    to_encode = ['sex', 'province', 'country', 'outcome']
    le = LabelEncoder()
    for i in range(len(to_encode)):
        data[to_encode[i]] = le.fit_transform(data[to_encode[i]].astype(str))

    X, y = data, data['outcome'].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)

    # classify with ADABoost
    ada_model = make_pipeline(
        AdaBoostClassifier()
    )

    ada_model.fit(X_train, y_train)
    print("Validation score (ADA):", ada_model.score(X_valid, y_valid)) 

    ada_pkl = '../models/ada_classifier.pkl'
    pickle.dump(ada_model, open(ada_pkl, 'wb'))

if __name__ == '__main__':
    main()