
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    input_data_file = "../dataset/2.0_cases_cleaned.csv.gz"
    data = pd.read_csv(input_data_file)

    to_encode = ['sex', 'province', 'country', 'outcome']
    le = LabelEncoder()
    for i in range(len(to_encode)):
        data[to_encode[i]] = le.fit_transform(data[to_encode[i]].astype(str))

    # train to test ratio -> 75:25 for MILESTONE 3 
    X, y = data.drop(['outcome'], axis = 1), pd.Series(data['outcome'].to_numpy())
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25)
    
    X_train_outputfile = "../dataset/3.1_X_train.csv.gz"
    X_valid_outputfile = "../dataset/3.1_X_valid.csv.gz"
    y_train_outputfile = "../dataset/3.1_y_train.csv.gz"
    y_valid_outputfile = "../dataset/3.1_y_valid.csv.gz"
    X_train.to_csv(X_train_outputfile, index = False, compression = 'gzip')
    X_valid.to_csv(X_valid_outputfile, index = False, compression = 'gzip')
    y_train.to_csv(y_train_outputfile, index = False, compression = 'gzip')
    y_valid.to_csv(y_valid_outputfile, index = False, compression = 'gzip')

if __name__ == '__main__':
    main()
