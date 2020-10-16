
import sys
import numpy as np
import pandas as pd


# Input/output data files
# input_file_cases = "processed_individual_cases_Sep20th2020.csv"
input_file_locations = "1.3_locations_removed_outliers.csv"
output_file_cases = "1.4_processed_location_Sep20th2020.csv"

# Read data from file
locations = pd.read_csv(input_file_locations)

US_data = locations[locations['Country_Region'] == 'US']
remaining = locations[locations['Country_Region'] != 'US']

grouped = US_data.groupby(['Province_State', 'Country_Region'])

aggreated = grouped.agg({
    'Last_Update': 'max', 
    'Lat': 'mean', 
    'Long_': 'mean', 
    'Confirmed': 'sum', 
    'Deaths': 'sum', 
    'Recovered': 'sum', 
    'Active': 'sum'
}).reset_index()

aggreated['Active'] = aggreated['Active'].apply(int)

aggreated['Combined_Key'] = aggreated['Province_State'] + ', ' + aggreated['Country_Region']

def newIncidenceRate(x):
    data = US_data[US_data['Province_State'] == x['Province_State']]
    data['Confirmed'] /= x['Confirmed']
    return np.sum(data['Confirmed'] * data['Incidence_Rate'])

aggreated['Incidence_Rate'] = aggreated.apply(
    newIncidenceRate, 
    axis = 1
)

aggreated['Case-Fatality_Ratio'] = aggreated['Deaths'] / aggreated['Confirmed']

aggreated = aggreated[aggreated['Province_State'] != 'Recovered']


result_data = pd.concat([remaining, aggreated])

# print(result_data)

result_data.to_csv(output_file_cases, index = False)
