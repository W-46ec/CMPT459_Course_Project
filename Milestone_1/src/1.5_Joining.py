
import sys
import numpy as np
import pandas as pd

# Input/output data files
input_file_cases = "1.2_processed_individual_cases_Sep20th2020.csv.gz"
input_file_locations = "1.4_processed_location_Sep20th2020.csv"
output_file_cases = "1.5_joined_individual_cases_Sep20th2020.csv.gz"

individual_cases = pd.read_csv(input_file_cases)
locations = pd.read_csv(input_file_locations)


individual_cases = individual_cases.merge(
    locations, 
    left_on = ['province', 'country'], 
    right_on = ['Province_State', 'Country_Region']
)

individual_cases.to_csv(output_file_cases, index = False, compression = 'gzip')
