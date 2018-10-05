import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

import seaborn as sns
plt.style.use('seaborn')

# DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'
DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'


# Read Data

full_data = pd.read_csv(DATA_PATH + 'clean_data.csv', index_col=0)
master = pd.read_csv(DATA_PATH + 'clean_data.csv',
                     usecols=['5_digit_NTD_ID',
                              # 'Agency',
                              'Modes',
                              # 'HQ_City',
                              'HQ_State',
                              # 'UZA',
                              # 'UZA_Name',
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
# Extract Unlinked Passenger Trips (UPT) to create target

UPT_cols = []

for col in full_data.columns:
    if col.endswith('UPT'):
        UPT_cols.append(col)

df_UPT = full_data[UPT_cols]

# Interopalte missing values
# df_UPT = df_UPT.dropna(thresh=60)
df_UPT = df_UPT[UPT_cols[3:]].interpolate(axis=1,
                                              limit=None,
                                              limit_direction='both')
# Add Agency ID Bck
df_UPT['5_digit_NTD_ID'] = full_data['5_digit_NTD_ID']

# Combine Ridership by Agency
df_UPT = df_UPT.groupby('5_digit_NTD_ID').sum()

# ### Create target ####
# Compare average total ridership between 2007 - 2014 (about the average peak)
# to 2014 - 2017. If ridership was greater than 95%, considered stable or
# increaseing [1], if recent ridership is less than 90% of past ridership,
# considered decreasing [0]

# Initial annual ridership, 2007-2014

init_cols = []
years_init = ['07', '08', '09', '10', '11', '12', '13', '14']

for year in years_init:
    for col in df_UPT.columns:
        if year in col:
            init_cols.append(col)

final_cols = []
years_final = ['15', '16', '17']

for year in years_final:
    for col in df_UPT.columns:
        if year in col:
            final_cols.append(col)


df_UPT['initial_ridership'] = df_UPT[init_cols].sum(axis=1)
df_UPT['initial_ridership'] = df_UPT['initial_ridership'].\
                              apply(lambda x: x / len(years_init))

df_UPT['recent_ridership'] = df_UPT[final_cols].sum(axis=1)
df_UPT['recent_ridership'] = df_UPT['recent_ridership'].\
                              apply(lambda x: x / len(years_final))

df_UPT['ridership_ratio'] = df_UPT['recent_ridership'] /\
                                  df_UPT['initial_ridership']
df_UPT['target'] = df_UPT['ridership_ratio'].\
                   apply(lambda x : 1 if x >= 0.95 else 0)

df_UPT['target'].describe()

df_UPT.head()

# ### Create Master Features ####
master.shape
master = master.fillna(0)
# Encode transit modes
master = pd.concat([master.drop('Modes', axis=1),
                    pd.get_dummies(master.Modes, prefix='mode_')], axis=1)

master = pd.concat([master.drop('HQ_State', axis=1),
                    pd.get_dummies(master.HQ_State)], axis=1)

# master.columns

grouped = master.groupby('5_digit_NTD_ID')
# grouped.describe
sum_cols = ['Passenger_Miles_FY',
            'Unlinked_Passenger_Trips_FY',
            'Fares',
            'Operating_Expenses_FY']

# for col in full_data.columns:
#     if col.endswith('UPT'):
#         sum_cols.append(col)

agency_cols = ['UZA_Area_SQ_Miles',
               'UZA_Population',
               'Service_Area_SQ_Miles',
               'Service_Area_Population']
master.columns
encoded_cols = ['mode__AG', 'mode__AR', 'mode__CB', 'mode__CC', 'mode__CR',
                'mode__DR', 'mode__DT', 'mode__FB', 'mode__HR', 'mode__IP',
                'mode__LR', 'mode__MB', 'mode__MG', 'mode__MO', 'mode__OR',
                'mode__PB', 'mode__RB', 'mode__SR', 'mode__TB', 'mode__TR',
                'mode__VP', 'mode__YR', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO',
                'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN',
                'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS',
                'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']

funcs = defaultdict()

for col in master.columns:
    if col in sum_cols:
        funcs[col] = np.sum
    elif col in agency_cols:
        funcs[col] = stats.mode
    elif col in encoded_cols:
        funcs[col] = np.min

master = grouped.agg(funcs).fillna(0)

for col in agency_cols:
    master[col] = master[col].apply(lambda x: x.mode[0])

temp = pd.DataFrame(df_UPT.target)

master.shape
temp.shape

master = master.join(temp, how='left')

X = master.drop('target', axis=1)
y = master.target

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)


rf = RandomForestClassifier()

rf.fit(train_X, train_y)
rf.score(test_X, test_y)
