import pandas as pd
import numpy as np
from scipy import stats
import gc
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from pandas.plotting import scatter_matrix
import seaborn as sns
plt.style.use('seaborn')

DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'

# DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'

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

# Objective # 2 - Is ridership increasing or decreasing?

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
plot_total_ridership(df_UPT_by_type, freq='quarterly')

# Try interpolating missing values. Don't interpolate if more than 5 years
# are missing to avoid over-estimating. Aligns with APTA estimates
df_UPT = df_UPT.dropna(thresh=60)
df_UPT_int = df_UPT[UPT_cols[3:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
df_UPT_int['Modes'] = df_UPT['Modes']
df_UPT_by_type = df_UPT_int.groupby('Modes').sum()
plot_total_ridership(df_UPT_by_type.T, freq='yearly')

# 2.2 Break down total annual ridership by mode
df_UPT_by_type['Total'] = df_UPT_by_type.sum(axis=1)
df_UPT_by_type = df_UPT_by_type.sort_values(by='Total', ascending=False)

# Plot monthly ridership of 5 most common modes, without total column
fig = df_UPT_by_type.T.iloc[:-1, 0:5].plot(figsize=(12, 12))
fig.set_title('Total Ridership in the United States by Type',
              fontsize=16)
fig.set_ylabel('Total Trips per Year', fontsize=14)
plt.show()

# 2.3 Investigate relationship between changes in total annual ridership and
# population served by agency

# Aggregate by transit agency ID, Create population bin feature

# Select columns, interpolate missing values
UPT_cols = ['5_digit_NTD_ID',
            'UZA_Area_SQ_Miles',
            'UZA_Population',
            'Service_Area_SQ_Miles',
            'Service_Area_Population']

for col in full_data.columns:
    if col.endswith('UPT'):
        UPT_cols.append(col)

df_UPT = full_data[UPT_cols]

df_UPT = df_UPT.dropna(thresh=60)
df_UPT_int = df_UPT[UPT_cols[5:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
df_UPT_int[UPT_cols] = df_UPT[UPT_cols]

# Agency properties to be aggregated by mode, ridership columns by sum

grouped = df_UPT_int.groupby('5_digit_NTD_ID')

sum_cols = []

for col in full_data.columns:
    if col.endswith('UPT'):
        sum_cols.append(col)

agency_cols = ['UZA_Area_SQ_Miles',
               'UZA_Population',
               'Service_Area_SQ_Miles',
               'Service_Area_Population']

funcs = defaultdict()

for col in df_UPT_int.columns:
    if col in sum_cols:
        funcs[col] = np.sum
    elif col in agency_cols:
        funcs[col] = stats.mode

df_areas = grouped.agg(funcs)

for col in agency_cols:
    df_areas[col] = df_areas[col].apply(lambda x: x.mode[0])

# Plot area and poluation served bins
n_bins = 20
n_ad_bins = 4

fig, axes = plt.subplots(2, 2, sharey=True, figsize=(12, 12))
axes[0, 0].hist(df_areas.UZA_Area_SQ_Miles, bins=n_bins)
axes[0, 0].set_title('UZA Area', fontsize=12)
axes[0, 0].set_ylabel('Number of Agencies', fontsize=12)
axes[0, 0].set_xlabel('Square Miles', fontsize=12)

axes[0, 1].hist(df_areas.Service_Area_SQ_Miles, bins=n_bins)
axes[0, 1].set_title('Service Area', fontsize=12)
axes[0, 1].set_xlabel('Square Miles', fontsize=12)

axes[1, 0].hist(df_areas.UZA_Population / 1e6, bins=n_bins)
axes[1, 0].set_ylabel('Number of Agencies', fontsize=12)
axes[1, 0].set_title('UZA Population', fontsize=12)
axes[1, 0].set_xlabel('Population (millions)', fontsize=12)

axes[1, 1].hist(df_areas.Service_Area_Population / 1e6,  bins=n_bins)
axes[1, 1].set_title('Service Area Population', fontsize=12)
axes[1, 1].set_xlabel('Population (millions)', fontsize=12)
fig.suptitle('Distribution of Area and Population Sizes Served',
             fontsize=14,
             y=.92)
plt.show()

for col in agency_cols:
    df_areas[col + '_bin'] = pd.cut(df_areas[col],
                                    bins=n_ad_bins,
                                    labels=np.arange(0, n_ad_bins))

for i, col in enumerate(agency_cols):
    # _, bin_arr = pd.cut(df_areas[col], bins=n_bins, retbins=True)
    _, bin_arr = pd.qcut(df_areas[col], q=n_ad_bins, retbins=True)
    legend = defaultdict()

    if i % 2 == 0:
        s = 'Area'
    else:
        s = 'Population'

    for j in range(0, n_ad_bins):
        if j == 0:
            legend[j] = (s + ' < ' + str(int(round(bin_arr[j + 1], -2))))
        elif j != n_ad_bins - 1:
            legend[j] = (str(int(round(bin_arr[j], -2))) +
                         ' < ' + s + ' < ' +
                         str(int(round(bin_arr[j + 1], -2))))
        else:
            legend[j] = (s + ' > ' + str(int(round(bin_arr[j], -2))))

    df = df_areas[[*sum_cols, col + '_bin']].\
                   groupby(col + '_bin').agg('sum').rename(index=legend)
    df.T.plot(figsize=(12, 12)).legend(loc='best')
    plt.title(col, fontsize=14)
    plt.ylabel('Total Annual Rides', fontsize=12)
    plt.show()

# 2.4 Investigate relationship between changes in total annual ridership and
# fares, costs and trip length

# Select columns, interpolate missing values
UPT_cols = ['5_digit_NTD_ID',
            'Unlinked_Passenger_Trips_FY',
            'Fares_FY',
            'Operating_Expenses_FY',
            'Average_Trip_Length_FY',
            'Average_Cost_per_Trip_FY',
            'Average_Fares_per_Trip_FY']

for col in full_data.columns:
    if col.endswith('UPT'):
        UPT_cols.append(col)

df_UPT = full_data[UPT_cols]

df_UPT = df_UPT.dropna(thresh=60)
df_UPT_int = df_UPT[UPT_cols[7:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
df_UPT_int[UPT_cols] = df_UPT[UPT_cols]

df_UPT_int['Margin'] = df_UPT_int['Average_Fares_per_Trip_FY'] -\
                                  df_UPT_int['Average_Cost_per_Trip_FY']
# Plot the distribution of margins
df = pd.DataFrame()
df['pos'] = df_UPT_int['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x < 0 else x)
df['neg'] = df_UPT_int['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x > 0 else x)
plt.bar(np.arange(0, df_UPT_int.shape[0], 1), df['pos'], color='g', width=1)
plt.bar(np.arange(0, df_UPT_int.shape[0], 1), df['neg'], color='r', width=1)
plt.ylim(-100, 20)
plt.ylabel('Net Revenue per Trip ($)')
plt.tick_params(bottom=False, labelbottom=False)
plt.title('Distribution of net Revenue per Trip')
plt.show()

df_UPT_by_agency = df_UPT_int.groupby('5_digit_NTD_ID').agg('sum')
df_UPT_by_agency.drop('Average_Trip_Length_FY', axis=1, inplace=True)
df_UPT_by_agency['Average_Cost_per_Trip_FY'] =\
                 df_UPT_by_agency.Operating_Expenses_FY /\
                 df_UPT_by_agency.Unlinked_Passenger_Trips_FY
df_UPT_by_agency['Average_Fares_per_Trip_FY'] =\
                 df_UPT_by_agency.Fares_FY /\
                 df_UPT_by_agency.Unlinked_Passenger_Trips_FY
df_UPT_by_agency['Margin'] = df_UPT_by_agency['Average_Fares_per_Trip_FY'] -\
                                  df_UPT_by_agency['Average_Cost_per_Trip_FY']
df_UPT_by_agency.head()

df = pd.DataFrame()
df['pos'] = df_UPT_by_agency['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x < 0 else x)
df['neg'] = df_UPT_by_agency['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x > 0 else x)
plt.bar(np.arange(0, df_UPT_by_agency.shape[0], 1), df['pos'],
        color='g', width=1)
plt.bar(np.arange(0, df_UPT_by_agency.shape[0], 1), df['neg'],
        color='r', width=1)
plt.ylim(-100, 20)
plt.ylabel('Net Revenue per Trip ($)')
plt.tick_params(bottom=False, labelbottom=False)
plt.title('Distribution of net Revenue per Trip by Agency')
plt.show()



# 2.5 Investigate the target variable - increase or decrease in ridership

# Objective 3: Create a US map of transit agenceis

# from mpl_toolkits.basemap import Basemap
# from geopy.geocoders import Nominatim
