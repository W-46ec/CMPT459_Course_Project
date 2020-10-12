import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
sns.set()

def main():
    input_file = sys.argv[1]
    input_file_2 = sys.argv[2]
    
    cases = pd.read_csv(input_file)
    locations = pd.read_csv(input_file_2)
    
    # 1.1 Exploratory Data Analysis
    # might need to make a 'plot' for extra visualization
    cases_missing_values = cases.isnull().sum()
    cases_missing_percent = cases.isnull().sum() * 100 / len(cases) 
    cases_missing_attrib = pd.DataFrame({'missing_values': cases_missing_values,
                                     'percent_missing': cases_missing_percent})
    
    cases_missing_attrib.sort_values('missing_values', inplace=True, ascending=False)
    print('Missing Value Data for [processed_individual_cases_Sep20th2020.csv]')
    print(cases_missing_attrib)
    
    loc_missing_values = locations.isnull().sum()
    loc_missing_percent = locations.isnull().sum() * 100 / len(cases) 
    loc_missing_attrib = pd.DataFrame({'missing_values': loc_missing_values,
                                     'percent_missing': loc_missing_percent})
    
    loc_missing_attrib.sort_values('missing_values', inplace=True, ascending=False)
    print('\nMissing Value Data for [processed_location_Sep20th2020.csv]')
    print(loc_missing_attrib)
    
    


if __name__ == '__main__':
    main()
    