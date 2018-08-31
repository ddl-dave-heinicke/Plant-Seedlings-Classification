import os
import pandas as pd
import gc
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from pandas.plotting import scatter_matrix
import seaborn as sns

os.chdir('C:\\Users\\Dave\\Documents\\Python Scripts\\Transit')

gc.collect()

full_data = pd.read_csv('clean_data.csv', index_col=0)
master = pd.read_csv('clean_data.csv', usecols=['5_digit_NTD_ID',
                                                'Agency',
                                                'Modes',
                                                'HQ_City',
                                                'HQ_State',
                                                'UZA',
                                                'UZA_Name',
                                                'UZA_Area_SQ_Miles',
                                                'UZA_Population',
                                                'Service_Area_SQ_Miles',
                                                'Service_Area_Population',
                                                'Passenger_Miles_FY',
                                                'Unlinked_Passenger_Trips_FY',
                                                'Average_Trip_Length_FY',
                                                'Fares_FY',
                                                'Operating_Expenses_FY',
                                                'Average_Cost_per_Trip_FY',
                                                'Average_Fares_per_Trip_FY'
                                                ])
print(master.head())

a = master.loc[master['5_digit_NTD_ID']]

# Create useful dictionaries

# Agency name by Agency ID
agencies = OrderedDict()

for row in master.iterrows():
    agencies[row[1][0]] = row[1][1]

# UZA Name by UZA ID
UZAs = OrderedDict()

for row in master.iterrows():
    UZAs[row[1][5]] = row[1][6]

# Agency IDs by UZA
agency_ID_by_UZA = defaultdict(set)

for row in master.iterrows():
    agency_ID_by_UZA[row[1][5]].add(row[1][0])

# UZA Name by Agency ID
agency_UZA = OrderedDict()

for row in master.iterrows():
    agency_UZA[row[1][0]] = row[1][6]

# Question - which UZAs have multiple transit agencies?
for UZA, Agencies in agency_ID_by_UZA.items():
    if len(Agencies) > 1:
        print('%-*s %i transit agencies' % (40, UZAs[UZA], len(Agencies)))

# Objective 1: Explore relationship between Service Area, Service Area
# Population, Number of Passenger Trips, and Passenger Miles

# For EDA, drop entries where service area info, passenger trip length or
# passenger miles are missing

EDA_1 = master[['5_digit_NTD_ID',
                'Agency',
                'UZA_Name',
                'Service_Area_SQ_Miles',
                'Service_Area_Population',
                'Passenger_Miles_FY',
                'Unlinked_Passenger_Trips_FY',
                'Average_Trip_Length_FY']]

print(EDA_1.shape)

EDA_1.dropna(axis=0, how='any', inplace=True)

print(EDA_1.shape)

print('There are %i Transit Agencies' % (len(set(EDA_1['5_digit_NTD_ID']))))

# Combine Transit Modes by Transit Agency

temp = EDA_1[['5_digit_NTD_ID',
              'Agency',
              'UZA_Name',
              'Service_Area_SQ_Miles',
              'Service_Area_Population']]\
            .groupby('5_digit_NTD_ID').first().reset_index(drop=True)

EDA_1_by_agency = EDA_1[['5_digit_NTD_ID',
                         'Passenger_Miles_FY',
                         'Unlinked_Passenger_Trips_FY',
                         'Average_Trip_Length_FY']]\
                         .groupby('5_digit_NTD_ID')\
                         .agg('sum')\
                         .reset_index(drop=True)

EDA_1_by_agency = temp.merge(EDA_1_by_agency,
                             left_index=True,
                             right_index=True)

del temp
gc.collect()

EDA_1_by_agency['Service_Area_Pop_Density'] =\
                EDA_1_by_agency.Service_Area_Population / \
                EDA_1_by_agency.Service_Area_SQ_Miles

# Stockton, Ca??? WTF?
EDA_1_by_agency.drop(EDA_1_by_agency.Service_Area_Pop_Density.
                     idxmax(), axis=0, inplace=True)

# Ok lets drop NYC
# EDA_1_by_agency.drop(EDA_1_by_agency.Unlinked_Passenger_Trips_FY.
#                     idxmax(), axis=0, inplace=True)

# Drop Population Outliers?
max_pop = 1000000

temp = EDA_1_by_agency.loc[EDA_1_by_agency.Service_Area_Population < max_pop]

temp.loc[temp.Service_Area_Pop_Density.idmax()]

temp.sort_values(by='Service_Area_Pop_Density', ascending=False).head(10)

plt.hist(temp.Service_Area_Population, bins=50)
plt.title('Servica Area Population Distribution')
plt.xlabel('Service Area Size')
plt.ylabel('Number of Service Areas')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Unlinked_Passenger_Trips_FY)
plt.title('Population Density vs Passenger Trips')
plt.xlabel('Service Area Population Density')
plt.ylabel('Unlinked Passenger Trips per Year')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Passenger_Miles_FY)
plt.title('Population Density vs Passenger Miles')
plt.xlabel('Service Area Population Density')
plt.ylabel('Passenger Miles')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Average_Trip_Length_FY)
plt.title('Population Density vs Passenger Miles')
plt.xlabel('Service Area Population Density')
plt.ylabel('Passenger Miles')
plt.show()

scatter_matrix(temp, diagonal='kde', figsize=(14, 14))
plt.show()

sns.heatmap(temp, annot=True)

gc.collect()
EDA_1_by_agency.loc[EDA_1_by_agency.Passenger_Miles_FY.idxmax()]
EDA_1_by_agency.loc[EDA_1_by_agency.Service_Area_Pop_Density.idxmax()]

EDA_1_by_agency[['Agency', 'UZA_Name', 'Service_Area_Pop_Density']]\
                 .sort_values(by=['Service_Area_Pop_Density'],
                              axis=0, ascending=False).head(20)

master.columns
