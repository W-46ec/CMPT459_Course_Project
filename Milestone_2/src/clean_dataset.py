import sys
import numpy as np
import pandas as pd

def main():
    txtfile = '../dataset/1.5_joined_individual_cases_Sep20th2020.csv.gz'
    data = pd.read_csv(txtfile)

    # prune redundant & irrelevant features
    data = data.drop(['Province_State', 'Country_Region', 'Lat', 'Long_'], axis = 1)
    data = data.drop(['additional_information', 'source', 'Last_Update', 'Combined_Key'], axis = 1)
    data = data.drop(['date_confirmation'], axis = 1)

    # Impute missing values for province
    data['province'][data['province'].isna()] = data['country'][data['province'].isna()]
    # data['province'] = data['province'].fillna('Unknown')
    data['sex'] = data['sex'].fillna('Unknown')

    # Impute missing values for age using random normal distribution
    age_mean = data[data['age'].notna()]['age'].mean()
    age_std = data[data['age'].notna()]['age'].std()
    imputed_age = np.random.normal(age_mean, age_std, len(data['age'].isna()))
    # Convert negative values to positive
    data['age'][data['age'].isna()] = imputed_age
    data['age'] = data['age'].apply(abs)

    # check NaN counts
    print('NaN count:')
    print(data.isna().sum())
    print(data.columns)
    
    output_file = "../dataset/2.0_cases_cleaned.csv.gz"
    # data.to_csv(output_file, index = False, compression = 'gzip')
    
if __name__ == '__main__':
    main()    