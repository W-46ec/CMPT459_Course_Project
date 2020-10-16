import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
sns.set()
#pd.set_option('display.max_rows', 190)

def main():
    #input_file = sys.argv[1]
    #input_file_2 = sys.argv[2]
    cases = pd.read_csv('processed_individual_cases_Sep20th2020.csv')
    locations = pd.read_csv('processed_location_Sep20th2020.csv')

    # 1.1 - Exploratory Data Analysis
    # Missing Value Counts
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
    # Various Feature Plots
    # cases - sex
    print('\nCreating pie plot for Cases - [Sex]...')
    new = cases
    new['sex'] = new['sex'].replace(np.nan, 'Unknown', regex=True)
    sex_labels = 'Unknown', 'Male', 'Female'
    sex_sizes = list(new['sex'].value_counts())
    sex_colors = ['violet', 'skyblue', 'lightcoral']
    plt.title('Invidual Cases - Percentage of [Sex]')
    plt.pie(sex_sizes, labels=sex_labels, colors=sex_colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.tight_layout()
    plt.savefig('plots/cases_sex.png')
    plt.clf()    
    print('Finished.')
    # cases - country
    print('Creating bar plot for Cases - [Country]...')
    new['country'] = new['country'].replace(np.nan, 'Unknown', regex=True) 
    plt2 = pd.value_counts(new['country']).plot.barh(figsize=(15,40), title='Individual Cases - [Country]', alpha=0.6, color=['teal', 'cyan'])   
    plt.tight_layout()
    plt.show()
    plt2.figure.savefig('plots/cases_country.png')
    plt.clf()
    print('Finished.')
    # cases - outcome
    print('Creating pie plot for Cases-[Outcome]...')
    outcome_labels = 'nonhospitalized', 'hospitalized', 'recovered', 'deceased'
    outcome_sizes = list(new['outcome'].value_counts())
    outcome_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    plt.title('Invidual Cases - Percentage of [Outcomes]')
    plt.pie(outcome_sizes, labels=outcome_labels, colors=outcome_colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.tight_layout()
    plt.savefig('plots/cases_outcome.png')
    plt.clf()
    print('Finished.')
    # locations - Country_Region
    print('Creating plot for Locations-[Country_Region]...')
    new = locations
    plt3 = pd.value_counts(new['Country_Region']).plot.barh(figsize=(15,40), title='Locations - [Country_Region]', alpha=0.6, color=['teal', 'cyan'])   
    plt.tight_layout()
    plt.show()
    plt3.figure.savefig('plots/locations_country.png')
    plt.clf()
    print('Finished.')
    # locations - Confirmed / Deaths / Recovered / Active
    print('Aggregating data for Locations-[Confirmed/Deaths/Recovered/Active]')
    loc_cases = locations.groupby(['Country_Region'])[["Confirmed", "Deaths", "Recovered", "Active"]].sum()
    loc_cases.sort_values("Confirmed", inplace=True, ascending=False)
    loc_cases.to_csv('plots/location_CasesByCountry.csv')
    print('Finished.')
    
    # lat/lon scatter plots take a while to generate, so I put them at the bottom
    
    # locations - latitude/longitude
    print('Creating scatter plot for Locations-[Lat/Lon]...')
    latlon_loc = locations.filter(['Lat', 'Long_'], axis=1)
    latlon_loc = latlon_loc.dropna()
    def get_clusters_loc(X):
        model = make_pipeline(
            # 188 unique countries in this dataset
            # print(new['country'].nunique())
            # 188/2 = 94
            KMeans(n_clusters=94, algorithm='elkan')
        )
        model.fit(X)
        return model.predict(X)    
    coord_clusters_loc = get_clusters_loc(latlon_loc)
    plt.title('Locations Dataset - Latitude/Longitude Visualization')
    plt.scatter(latlon_loc['Long_'], latlon_loc['Lat'], marker='o', c=coord_clusters_loc, cmap='Dark2', alpha=0.5, s=4)
    plt.tight_layout()
    plt.savefig('plots/locations_coordinate_clusters.png')
    plt.clf()
    print('Finished.')
    # cases - lat/lon
    print('Creating scatter plot for Cases-[Lat/Lon]...')
    latlon = cases.filter(['latitude', 'longitude'], axis=1)
    latlon = latlon.dropna()
    def get_clusters_cases(X):
        model = make_pipeline(
            # 136 unique countries in this dataset
            # print(new['country'].nunique())
            # 136/2 = 68
            KMeans(n_clusters=68, algorithm='auto')
        )
        model.fit(X)
        return model.predict(X)    
    coord_clusters = get_clusters_cases(latlon)
    plt.title('Individual Cases - Latitude/Longitude Visualization')
    plt.scatter(latlon['longitude'], latlon['latitude'], marker='o', c=coord_clusters, cmap='Dark2', alpha=0.5, s=4)
    plt.tight_layout()
    plt.savefig('plots/cases_coordinate_clusters.png')
    plt.clf()
    print('Finished.')

if __name__ == '__main__':
    main()
    