import sys
import numpy as np
import pandas as pd

def main():
    txtfile = '../dataset/1.5_joined_individual_cases_Sep20th2020.csv.gz'
    data = pd.read_csv(txtfile)
    
    # prune redundant & irrelevant features
    data = data.drop(['Province_State', 'Country_Region', 'Lat', 'Long_'], axis=1)
    data = data.drop(['additional_information', 'source', 'Last_Update', 'Combined_Key'], axis=1)
    
    # remove this section when the merge is fixed
    data['province'] = data['province'].fillna('Unknown')
    
    # check NaN counts
    print('NaN count:')
    print(data.isna().sum())
    print(data.columns)
    
    output_file = "../dataset/2.0_cases_cleaned.csv.gz"
    data.to_csv(output_file, index = False, compression = 'gzip')
    
if __name__ == '__main__':
    main()    