import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats
sns.set()



def main():
    # import 'locations' dataset
    cases = pd.read_csv('1.2_processed_individual_cases_Sep20th2020.csv.gz')
    locations = pd.read_csv('processed_location_Sep20th2020.csv')

    # 1.3 - Outlier detection and correction
    
    # Locations
    # Using z-score, remove outliers from 'Incidence_Rate' and 'Case-Fatality_Ratio'
    locations = locations.dropna(subset=['Incidence_Rate', 'Case-Fatality_Ratio'])
    incidence_zs = np.abs(stats.zscore(locations['Incidence_Rate']))
    crossfatal_zs = np.abs(stats.zscore(locations['Case-Fatality_Ratio']))
    locations['incidence_zscore'] = incidence_zs
    locations['crossfatal_zscore'] = crossfatal_zs
    
    # filter entries with zscore greater than 3
    locations = locations[locations['incidence_zscore'] < 3]
    locations = locations[locations['crossfatal_zscore'] < 3]
    
    # remove calculation columns and export to csv
    locations = locations.drop(columns=['incidence_zscore', 'crossfatal_zscore'])
    locations.to_csv('1.3_locations_removed_outliers.csv', index = False)
    
    # Cases
    # can't remove outliers for 'age' unless we drop half the dataset due to empty elements
    '''
    cases = cases.dropna(subset=['age'])
    print(len(cases))
    age_zs = np.abs(stats.zscore(cases['age']))
    cases['age_zscore'] = age_zs
    print(cases['age_zscore'])
    cases = cases[cases['age_zscore'] < 3]
    print(len(cases))
    cases.drop(columns=['age_zscore'])
    cases.to_csv('1.3_cases_removed_outliers.csv.gz', index = False, compression = 'gzip')    
    '''
    
if __name__ == '__main__':
    main()
    