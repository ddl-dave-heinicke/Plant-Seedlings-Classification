# Preprocessor
import pandas as pd
import os

DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\input_data\\'
OUTPUT_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'

dtypes = {'5 digit NTD ID': 'str'}

# Create dataframe of dataframes
df_dict = {f.split('.')[0]: pd.read_csv(os.path.join(DATA_PATH, f),
                                        dtype=dtypes, thousands=',')
           for f in os.listdir(DATA_PATH)
           if f.endswith('.csv')}

# Iterate through data files from APTA & FTA
for k in df_dict.keys():

    # Claen up column names
    if 'Mode' in df_dict[k].columns:
        df_dict[k].rename(columns={'Mode': 'Modes'}, inplace=True)
    for col in df_dict[k].columns:
        df_dict[k].rename(columns={col: col.lstrip()}, inplace=True)
    for col in df_dict[k].columns:
        df_dict[k].rename(columns={col: col.rstrip()}, inplace=True)
    for col in df_dict[k].columns:
        df_dict[k].rename(columns={col: col.replace(' ', '_')}, inplace=True)

    # Drop 'summary statistics' rows at end in transit data
    for index, row in df_dict[k].iterrows():
        if index > 2121:
            df_dict[k] = df_dict[k].drop(index, axis=0)

    # Create unique ID for each row to merge on later
    # AgencyID + Transit Mode Code + Type of Service Code
    df_dict[k]['Unique_ID'] = df_dict[k][['5_digit_NTD_ID', 'Modes', 'TOS']].\
                                          apply(lambda x: '_'.join(x), axis=1)
    df_dict[k].set_index('Unique_ID')
    df_dict[k].drop('Unique_ID', axis=1, inplace=True)

    # Drop unnecessary columns
    df_dict[k].drop('4_digit_NTD_ID', axis=1, inplace=True)

    # Clean up columns in 'Master' sheet with dollar signs
    dollar_cols = ['Fares_FY',
                   'Operating_Expenses_FY',
                   'Average_Cost_per_Trip_FY',
                   'Average_Fares_per_Trip_FY']

    if k == 'Master':
        for col in dollar_cols:
            df_dict['Master'][col].fillna('0', inplace=True)
            df_dict['Master'][col] = df_dict['Master'][col].\
                                             apply(lambda x: x.replace('-', '0'))
            df_dict['Master'][col] = df_dict['Master'][col]\
                                             .apply(lambda x: x.lstrip().
                                             lstrip('$').
                                             rstrip().
                                             replace(',',''))
# Convert column data types
for col in df_dict['Master'].columns:
    df_dict['Master']['UZA'].astype('int', errors='raise')  # UZA ID
    pd.to_numeric(df_dict['Master']['UZA_Area_SQ_Miles'])
    pd.to_numeric(df_dict['Master']['UZA_Population'])
    pd.to_numeric(df_dict['Master']['Service_Area_SQ_Miles'])
    pd.to_numeric(df_dict['Master']['Service_Area_Population'])
    # Make date features categorical
    df_dict['Master']['Most_Recent_Report_Year'].astype('object',
                                                        errors='raise')
    df_dict['Master']['FY_End_Month'].astype('object', errors='raise')
    df_dict['Master']['FY_End_Year'].astype('object', errors='raise')

# Convert cost/budget columns to floats
for col in dollar_cols:
    df_dict['Master'][col] = df_dict['Master'][col].astype('float',
                                                           errors='raise')

# Merge into single data frame
full_df = pd.DataFrame(df_dict['Master'])
# Merge columns with monthly ridership data
cols_to_merge = df_dict['UPT_monthly'].columns[9:]

# Clean up column names
for k in df_dict.keys():
    if k != 'Master':
        for col in cols_to_merge:
            df_dict[k].rename(columns={col: col + '_' + k.split('_')[0]},
                              inplace=True)

for k in list(df_dict.keys())[1:]:
    cols_to_merge = df_dict[k].columns[9:]
    full_df = full_df.join(df_dict[k][cols_to_merge])

# full_df.shape

# full_df.info()

# full_df.head()

# Save clean data
full_df.to_csv(OUTPUT_PATH + 'clean_data_test.csv')
