import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

import seaborn as sns
plt.style.use('seaborn')

# DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'
DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'

# ROC curve
def plot_roc_curve(test_y, preds_proba):
    preds_proba = preds_proba[:,1]
    fpr, tpr, thresholds = roc_curve(test_y, preds_proba)

    plt.plot(fpr, tpr, color='b',
             # label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

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

master.columns

grouped = master.groupby('5_digit_NTD_ID')
# grouped.describe
sum_cols = ['Passenger_Miles_FY',
            'Unlinked_Passenger_Trips_FY',
            'Fares_FY',
            'Operating_Expenses_FY']

agency_cols = ['UZA_Area_SQ_Miles',
               'UZA_Population',
               'Service_Area_SQ_Miles',
               'Service_Area_Population']

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
        funcs[col] = np.max

master = grouped.agg(funcs).fillna(0)

for col in agency_cols:
    master[col] = master[col].apply(lambda x: x.mode[0])

temp = pd.DataFrame(df_UPT.target)

master = master.join(temp, how='left')

# Feature Extraction
def divide_with_zeros(a, b):
    if b == 0:
        return np.nan
    else:
        return (a/b)

master['trips_per_mile'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Unlinked_Passenger_Trips_FY'],
                                    row['Passenger_Miles_FY']),
                                    axis=1)

master['fares_per_mile'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Fares_FY'],
                                    row['Passenger_Miles_FY']),
                                    axis=1)

master['cost_per_mile'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Operating_Expenses_FY'],
                                    row['Passenger_Miles_FY']),
                                    axis=1)

master['miles_per_trip'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Passenger_Miles_FY'],
                                    row['Unlinked_Passenger_Trips_FY']),
                                    axis=1)

master['fare_per_trip'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Fares_FY'],
                                    row['Unlinked_Passenger_Trips_FY']),
                                    axis=1)

master['cost_per_trip'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['Operating_Expenses_FY'],
                                    row['Unlinked_Passenger_Trips_FY']),
                                    axis=1)

master['net_per_trip'] = master['cost_per_trip'] - master['fare_per_trip']

master['net_per_mile'] = master['cost_per_mile'] - master['fares_per_mile']

master['net_revenue'] = master['Fares_FY'] - master['Operating_Expenses_FY']

master['UZA_pop_density'] = master.\
                            apply(lambda row:
                                  divide_with_zeros(
                                    row['UZA_Population'],
                                    row['UZA_Area_SQ_Miles']),
                                    axis=1)

master['service_area_pop_density'] = master.\
                                        apply(lambda row:
                                          divide_with_zeros(
                                            row['Service_Area_Population'],
                                            row['Service_Area_SQ_Miles']),
                                            axis=1)

master['service_to_uza_pop'] = master.\
                                    apply(lambda row:
                                          divide_with_zeros(
                                            row['Service_Area_Population'],
                                            row['UZA_Population']),
                                            axis=1)

master['service_to_uza_area'] = master.\
                                    apply(lambda row:
                                          divide_with_zeros(
                                            row['Service_Area_SQ_Miles'],
                                            row['UZA_Area_SQ_Miles']),
                                            axis=1)

master['cost_per_person'] = master.\
                                apply(lambda row:
                                      divide_with_zeros(
                                        row['Operating_Expenses_FY'],
                                        row['Service_Area_Population']),
                                        axis=1)

master = master.apply(lambda x: x.fillna(x.mean()), axis=0)

# master.describe()
# master.info()
# mode_cols = [col for col in master.columns if 'mode_' in col]
# mode_cols
# master['mode__SR'].describe()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


X = master.drop('target', axis=1)
y = master.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)


rf = RandomForestClassifier()

# Tune a lightgbm model

# dir(GridSearchCV)


lgb_clf = lgb.LGBMClassifier(n_estimators=1000,
                             objective='binary',
                             random_state=42,
                             eval_metric='roc_auc',
                             )

# lgb_clf.get_params().keys()

param_grid = {'num_leaves': [5, 10, 15, 20],
              'max_depth': [4, 5, 6],
              'learning_rate': [0.01, 0.1],
              'min_split_gain': [0],
              'min_child_weight': [3, 4, 5, 6, 7],
              'colsample_bytree': [1],
              'reg_alpha': [1, 0, 0.1, 0.01],
              'reg_lambda': [0],
              }

lgb_cv = GridSearchCV(lgb_clf,
                      param_grid=param_grid,
                      cv=4,
                      scoring='roc_auc',
                      verbose=1)

# help(lgb_clf)
# help(lgb_cv)


lgb_cv.fit(train_X, train_y)

lgb_cv.best_score_
lgb_cv.best_params_

preds = lgb_cv.best_estimator_.predict(test_X)
preds_proba = lgb_cv.best_estimator_.predict_proba(test_X)
score = lgb_cv.best_estimator_.score(test_X, test_y)

cm = confusion_matrix(test_y, preds)
print(cm)
print(classification_report(test_y, preds))
print(roc_auc_score(test_y, preds))

plot_roc_curve(test_y, preds_proba)

preds.mean()
y.describe()


# Small data! Logisitc Regression?

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000,
                        verbose=1)

params_lr = {'penalty': ['l1', 'l2'],
             'tol': [0.0001],
             'C':[3, 4, 5],
             'solver':['liblinear']}

lr_cv = GridSearchCV(lr,
                     param_grid=params_lr,
                     scoring='roc_auc',
                     cv=4,)

lr_cv.fit(train_X, train_y)

lr_cv.best_score_
lr_cv.best_params_

preds = lr_cv.best_estimator_.predict(test_X)
preds_proba = lr_cv.best_estimator_.predict_proba(test_X)
score = lr_cv.best_estimator_.score(test_X, test_y)
print(score)
cm = confusion_matrix(test_y, preds)
print(cm)
print(classification_report(test_y, preds))
print(roc_auc_score(test_y, preds))

plot_roc_curve(test_y, preds_proba)
