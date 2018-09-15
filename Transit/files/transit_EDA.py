import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from pandas.plotting import scatter_matrix
import seaborn as sns

# DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'

DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'

# Definitions


def plot_total_ridership(df, freq='monthly'):

    labels = []

    if freq == 'monthly':
        for i in range(2002, 2019):
            for j in range(1, 13):
                labels.append('%i, %i' % (j, i))
    elif freq == 'quarterly':
        for i in range(2002, 2019):
            for j in range(1, 5):
                labels.append('Q%i %i' % (j, i))
    elif freq == 'yearly':
        for i in range(2002, 2019):
                labels.append('%i' % (i))
    else:
        print('Invalid Split')
        return None

    freq_dict = {'monthly': [1],
                 'quarterly': 3,
                 'yearly': 12}

    df['Total'] = df.sum(axis=1)
    df = df.reset_index().drop('index', axis=1)
    df = df.groupby(df.index // freq_dict[freq] * freq_dict[freq]).sum()
    df['labels'] = labels[0: df.shape[0]]
    df.set_index('labels', drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax = df.iloc[0:-1].Total.plot()
    ax.set_title('%s Total Ridership in the United States' % freq.capitalize(),
                 fontsize=16)
    ax.set_ylabel('Total Trips per Year', fontsize=14)
    plt.show()

    return None


# Read Data

full_data = pd.read_csv(DATA_PATH + 'clean_data.csv', index_col=0)
master = pd.read_csv(DATA_PATH + 'clean_data.csv',
                     usecols=['5_digit_NTD_ID',
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

# Create Service Area Population Density Feature
EDA_1_by_agency['Service_Area_Pop_Density'] =\
                EDA_1_by_agency.Service_Area_Population / \
                EDA_1_by_agency.Service_Area_SQ_Miles

# Passenger Miles per Capita Feature
EDA_1_by_agency['Passenger_Miles_PC'] =\
                EDA_1_by_agency.Passenger_Miles_FY / \
                EDA_1_by_agency.Service_Area_Population

# Passenger Miles per Capita Feature
EDA_1_by_agency['Passenger_Trips_PC'] =\
                EDA_1_by_agency.Unlinked_Passenger_Trips_FY / \
                EDA_1_by_agency.Service_Area_Population

# Stockton, Ca??? Huh? Looks like a data entry error
EDA_1_by_agency.drop(EDA_1_by_agency.Service_Area_Pop_Density.
                     idxmax(), axis=0, inplace=True)

# Investigate outliers
EDA_1_by_agency.iloc[EDA_1_by_agency.Passenger_Trips_PC.idxmax()]

EDA_1_by_agency.loc[EDA_1_by_agency.UZA_Name.str.contains('KY')]

# Drop Population Outliers - only service areas between 50k and 1 million
min_pop = 0000
max_pop = 10000000
miles_cutoff = 10000000

temp =\
     EDA_1_by_agency.loc[(EDA_1_by_agency.Service_Area_Population < max_pop)
                         & (EDA_1_by_agency.Service_Area_Population > min_pop)
                         & (EDA_1_by_agency.Passenger_Miles_FY > miles_cutoff)]

temp.sort_values(by='Service_Area_Pop_Density', ascending=False).tail(10)

plt.hist(temp.Service_Area_Population, bins=50)
plt.title('Servica Area Population Distribution')
plt.xlabel('Service Area Population')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Passenger_Trips_PC)
plt.title('Population Density vs Passenger Trips')
plt.xlabel('Service Area Population Density')
plt.ylabel('Passenger Trips per Year per Capita')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Passenger_Miles_PC)
plt.title('Population Density vs Passenger Miles')
plt.xlabel('Service Area Population Density')
plt.ylabel('Passenger Miles per Year per Capita')
plt.show()

plt.scatter(temp.Service_Area_Pop_Density, temp.Average_Trip_Length_FY)
plt.title('Population Density vs Passenger Miles')
plt.xlabel('Service Area Population Density')
plt.ylabel('Average Trip Length (Miles)')
plt.show()

scatter_matrix(temp, diagonal='kde', figsize=(14, 14))
plt.show()

sns.heatmap(temp, annot=True)

gc.collect()

# Objective # 2 - Where is ridership increasing, and where is it decresing?

# 2.1 Check missing values

# Dataframe of unlinked passenger trips (UPT)
UPT_cols = ['5_digit_NTD_ID', 'Agency', 'Modes']

for col in full_data.columns:
    if col.endswith('UPT'):
        UPT_cols.append(col)

df_UPT = full_data[UPT_cols]

# Per Read Me: Data quality and completeness have improved significantly over
# time. The monthly data collection was introduced as a pilot program in 2002.
# Over time, most transit properties developed new internal data collection and
# processing methods to meet the new requirements. These developments, combined
# with the implementation of more sophisticated validation checks by FTA, have
# resulted in more complete and accurate data in more recent years.

# What happens when we ignore missing data and just plot ridership by mode
# as-is?

df_UPT_by_type = df_UPT[UPT_cols[2:]].groupby('Modes').sum().T
df_UPT_by_type = df_UPT_by_type.fillna(0)
# df_UPT_by_type.head()
plot_total_ridership(df_UPT_by_type, freq='quarterly')
df_UPT_by_type.head()
# Try interpolating missing values. Don't interpolate if more than 5 years
# are missing to avoid over-estimating. Aligns with APTA estimates

df_UPT_int = df_UPT[UPT_cols[3:]].dropna(thresh=60)
df_UPT_int = df_UPT_int.interpolate(axis=1, limit=None, limit_direction='both')
df_UPT_int['Modes'] = df_UPT['Modes']
df_UPT_by_type = df_UPT_int.groupby('Modes').sum().T

plot_total_ridership(df_UPT_by_type, freq='quarterly')

# 2.2 Break down total annual ridership by type

df_UPT_by_type['Total'] = df_UPT_by_type.sum(axis=1)
df_UPT_by_type = df_UPT_by_type.sort_values(by='Total', ascending=False)

fig, ax = plt.subplots(figsize=(14, 14))
ax = df_UPT_by_type.plot()
ax.set_title('Total Ridership in the United States by Type',
             fontsize=16)
ax.set_ylabel('Total Trips per Year', fontsize=14)
plt.show()

df_UPT_by_type.head(22)
df_UPT_by_type.shape

# Objective 3: Create a US map of transit agenceis

# from mpl_toolkits.basemap import Basemap
# from geopy.geocoders import Nominatim
