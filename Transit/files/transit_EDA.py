import pandas as pd
import numpy as np
from scipy import stats
import gc
from collections import defaultdict, OrderedDict

# Plotting
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
plt.style.use('seaborn')

DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'
PLOT_PATH = 'C:\\Users\\Dave\\Documents\\GitHub\\Sample-Work\\Transit\\plots\\'

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

    freq_dict = {'monthly': 1,
                 'quarterly': 3,
                 'yearly': 12}

    df['Total'] = df.sum(axis=1)
    df = df.reset_index().drop('index', axis=1)
    df = df.groupby(df.index // freq_dict[freq] * freq_dict[freq]).sum()
    df['labels'] = labels[0: df.shape[0]]
    df.set_index('labels', drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = df.iloc[0:-1].Total.plot()
    ax.set_title('%s Total Ridership in the United States' % freq.capitalize(),
                 fontsize=16)
    ax.set_ylabel('Total Trips per Year', fontsize=14)
    # plt.savefig(PLOT_PATH + 'quarterly_ridership.png')
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
                'UZA_Population',
                'Service_Area_SQ_Miles',
                'Service_Area_Population',
                'Passenger_Miles_FY',
                'Unlinked_Passenger_Trips_FY',
                'Average_Trip_Length_FY']]

EDA_1.dropna(axis=0, how='any', inplace=True)

print('There are %i Transit Agencies' % (len(set(EDA_1['5_digit_NTD_ID']))))

print('The agencies operate in %i states & terriories and %i municipalities.'
      % (len(np.unique(master.HQ_State)), len(np.unique(master.HQ_City))))

# Combine Transit Modes by Transit Agency

temp = EDA_1[['5_digit_NTD_ID',
              'Agency',
              'UZA_Name',
              'UZA_Population',
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

# Service area population in millions
EDA_1_by_agency['Service_Area_Pop_mill'] =\
                EDA_1_by_agency.Service_Area_Population / 1000000

EDA_1_by_agency['UZA_Pop_mill'] =\
                EDA_1_by_agency.UZA_Population / 1000000


# Check for obvious errors in the data
EDA_1_by_agency[['UZA_Name',
                 'Agency',
                 'Service_Area_Pop_Density',
                 'UZA_Pop_mill',
                 'Service_Area_Pop_mill',
                 'Service_Area_SQ_Miles']].\
                 loc[EDA_1_by_agency.UZA_Pop_mill < 100].\
                 sort_values(by='Service_Area_Pop_Density',
                             ascending=False)[:30]

# List of major errors noted and cleaned in the preprocessor:
# 1) Altamont Corridor Express: Service Area should include San Jose<->Stockton
# 2) Mecklenburg County DSS's service area is Charlotte's 688 sq mi, not 31
# 3) San Juan / Fajardo Ferry is unusual, but better described as serving 876.2
#    sq. miles (San Juan-Fajardo urban area)
# 4) The Detrit People Mover shows a very high population density, but
#    it really only serves central Detroit, so for purposes of the model its
#    probably accurate
# 5) Polk County Transit Services serves all of Polk Co (1798 sq mi)
# 6) University of Georgia Transit System serves all of Athens, 118 sq miles
# 7) Augusta Richmond County Transit Department serves 302 sq miles
# 9) Ventura Intercity Service Transit Authority serves ~800 sq.mi


# Optional - Drop Population Outliers
min_pop = 0000
max_pop = 7000000
miles_cutoff = 0

temp =\
     EDA_1_by_agency.loc[(EDA_1_by_agency.UZA_Population < max_pop)
                         & (EDA_1_by_agency.UZA_Population > min_pop)
                         & (EDA_1_by_agency.Passenger_Miles_FY > miles_cutoff)]

temp.sort_values(by='Passenger_Trips_PC', ascending=False).head(10)

# Distribution of Service Area Populations
plt.hist(temp.Service_Area_Population, bins=50)
plt.title('Servica Area Population Distribution')
plt.xlabel('Service Area Population')
# plt.savefig(PLOT_PATH + 'dist_agencies.png')
plt.show()

# Correlation bewteen population density and passenger trips per capita?
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(temp.Service_Area_Pop_Density, temp.Passenger_Trips_PC)
ax[0].set_title('Population Density vs Passenger Trips')
ax[0].set_xlabel('Service Area Population Density')
ax[0].set_ylabel('Passenger Trips per Year per Capita')
ax[0].set_xlim(0, 10000)

# Correlation bewteen population density and passenger miles per capita?
ax[1].scatter(temp.Service_Area_Pop_Density, temp.Passenger_Miles_PC)
ax[1].set_title('Population Density vs Passenger Miles')
ax[1].set_xlabel('Service Area Population Density')
ax[1].set_ylabel('Passenger Miles per Year per Capita')
ax[1].set_xlim(0, 10000)
ax[1].set_ylim(0, 500)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'pop_desnity.png')
plt.show()

# Correlation bewteen population density and trip length?
plt.scatter(temp.Service_Area_Pop_Density, temp.Average_Trip_Length_FY)
plt.title('Population Density vs Passenger Miles')
plt.xlabel('Service Area Population Density')
plt.ylabel('Average Trip Length (Miles)')
plt.xlim(0, 10000)
plt.show()

# Correlation between anything?
scatter_matrix(temp[['Service_Area_Population',
                     'Service_Area_Pop_Density',
                     'Passenger_Miles_FY',
                     'Unlinked_Passenger_Trips_FY',
                     'Average_Trip_Length_FY']],
               diagonal='kde', figsize=(14, 14))
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
plot_total_ridership(df_UPT_by_type, freq='monthly')

# Try interpolating missing values. Don't interpolate if more than 5 years
# are missing to avoid over-estimating. Aligns with APTA estimates
df_UPT = df_UPT.dropna(thresh=60)
df_UPT_int = df_UPT[UPT_cols[3:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
df_UPT_int['Modes'] = df_UPT['Modes']
df_UPT_by_type = df_UPT_int.groupby('Modes').sum()
plot_total_ridership(df_UPT_by_type.T, freq='quarterly')

# 2.2 Break down total annual ridership by mode
df_UPT_by_type['Total'] = df_UPT_by_type.sum(axis=1)
df_UPT_by_type = df_UPT_by_type.sort_values(by='Total', ascending=False)

# Plot monthly ridership of 5 most common modes, without total column
fig = df_UPT_by_type.T.iloc[:-1, 0:5].plot(figsize=(10, 10))
fig.set_title('Total Ridership in the United States by Type',
              fontsize=16)
fig.set_ylabel('Total Trips per Year', fontsize=14)
plt.tight_layout()
# plt.savefig(PLOT_PATH + 'ridership_by_type.png')
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
    df.T.plot(figsize=(10, 10)).legend(loc='best')
    plt.title(col, fontsize=14)
    plt.ylabel('Total Annual Rides', fontsize=12)
    plt.tight_layout()
    # plt.savefig(PLOT_PATH + col +'.png')
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

# Create revenue by trip data frame
df = pd.DataFrame()
df['pos'] = df_UPT_int['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x < 0 else x)
df['neg'] = df_UPT_int['Margin'].sort_values(ascending=False).\
                                 apply(lambda x: 0 if x > 0 else x)

# Plot revenue by trip
plt.bar(np.arange(0, df_UPT_int.shape[0], 1), df['pos'], color='g', width=1)
plt.bar(np.arange(0, df_UPT_int.shape[0], 1), df['neg'], color='r', width=1)
plt.ylim(-100, 20)
plt.ylabel('Net Revenue per Trip ($)')
plt.tick_params(bottom=False, labelbottom=False)
plt.title('Distribution of net Revenue per Trip')
plt.show()

# Create revenue by agency
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

# Which agencies do best and worst?
temp = df_UPT_by_agency[['Margin']].merge(master,
                                          how='left',
                                          left_index=True,
                                          right_on='5_digit_NTD_ID')

temp[['Agency', 'Margin']].sort_values(by='Margin',
                                       ascending=False).head(50)

temp[['Agency', 'Margin']].dropna().sort_values(by='Margin',
                                                ascending=False).tail(50)

# Plot revenue by agency
plt.bar(np.arange(0, df_UPT_by_agency.shape[0], 1), df['pos'],
        color='g', width=1)
plt.bar(np.arange(0, df_UPT_by_agency.shape[0], 1), df['neg'],
        color='r', width=1)
plt.ylim(-100, 20)
plt.ylabel('Net Revenue per Trip ($)')
plt.tick_params(bottom=False, labelbottom=False)
plt.title('Distribution of net Revenue per Trip by Agency')
plt.savefig(PLOT_PATH + 'revenue_by_agnecy.png')
plt.show()

# Do fares and operating costs affect chnage in ridership?

# Aggregate by transit agency ID, Create population bin feature

# Select columns, interpolate missing values
UPT_cols = ['5_digit_NTD_ID',
            'Average_Cost_per_Trip_FY',
            'Average_Fares_per_Trip_FY']

for col in full_data.columns:
    if col.endswith('UPT'):
        UPT_cols.append(col)

df_UPT = full_data[UPT_cols]

df_UPT = df_UPT.dropna(thresh=60)
df_UPT.columns
df_UPT_int = df_UPT[UPT_cols[3:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
df_UPT_int[UPT_cols] = df_UPT[UPT_cols]

# Agency properties to be aggregated by mode, ridership columns by sum

grouped = df_UPT_int.groupby('5_digit_NTD_ID')

sum_cols = []

for col in full_data.columns:
    if col.endswith('UPT'):
        sum_cols.append(col)

agency_cols = ['Average_Cost_per_Trip_FY',
               'Average_Fares_per_Trip_FY']

funcs = defaultdict()

for col in df_UPT_int.columns:
    if col in sum_cols:
        funcs[col] = np.sum
    elif col in agency_cols:
        funcs[col] = np.mean

df_costs = grouped.agg(funcs)
df_costs.head()

n_ad_bins = 4
bins = {}

for col in agency_cols:
    df_costs[col + '_bin'], bins[col] = pd.qcut(df_areas[col],
                                                q=n_ad_bins,
                                                labels=np.arange(0, n_ad_bins),
                                                retbins=True)

df_costs_agg = df_costs.groupby('Average_Cost_per_Trip_FY_bin').sum()

df_costs_agg.drop(['Average_Cost_per_Trip_FY',
                   'Average_Fares_per_Trip_FY'],
                  inplace=True, axis=1)

df_costs_agg = df_costs_agg.T

df_fares_agg = df_costs.groupby('Average_Fares_per_Trip_FY_bin').sum()
df_fares_agg.drop(['Average_Cost_per_Trip_FY',
                   'Average_Fares_per_Trip_FY'],
                  inplace=True, axis=1)

df_fares_agg = df_fares_agg.T

df_costs_agg.plot(figsize=(10,10))
plt.ylabel('Total Ridership per Month')
plt.title('Cost per Trip', fontsize=14)
plt.legend(['\$0-\$4.12', '\$4.12-\$13.45', '\$13.45-\$20.37', '\$20 +'])
plt.savefig(PLOT_PATH + 'cost_per_trip.png')
plt.show()

df_fares_agg.plot(figsize=(10,10))
plt.ylabel('Total Ridership per Month')
plt.title('Fare per Trip', fontsize=14)
plt.legend(['\$0-\$0.53', '\$.53-\$1.38', '\$1.38-\$2.32', '\$2.32-\$120'])
plt.savefig(PLOT_PATH + 'fare_per_trip.png')
plt.show()

bins

help(ax[0].set_xticks)
