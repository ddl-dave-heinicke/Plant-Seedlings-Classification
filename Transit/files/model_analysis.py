import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
from pdpbox import pdp

# eli5
import eli5
from eli5.sklearn import PermutationImportance

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# SHAP
import shap
shap.initjs()

DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'
# DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'


def shuffle_verify(X, y, model):

    shuffle_arr = np.arange(1, 11, 1)

    scores = []

    for rs in shuffle_arr:

        train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=rs)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        scores.append(roc_auc_score(test_y, preds))

    return(np.mean(scores))


def shuffle_SHAP(X, y, model, n_shuffles=10):

    shuffle_arr = np.arange(1, n_shuffles+1, 1)

    feature_names = X.columns.tolist()
    feat_cols = ['m_s_' + str(i) for i in range(1, n_shuffles+1)]
    feat_means = pd.DataFrame(columns=feat_cols, index=feature_names)

    for rs in shuffle_arr:
        train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=rs)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_X)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        for col in shap_df.columns:
            temp = shap_df[col].apply(lambda x: abs(x))
            # feat_means[str('m_s_' + rs)] = np.nan
            feat_means['m_s_' + str(rs)].loc[col] = np.mean(temp)

    feat_means['total'] = feat_means.sum(axis=1)

    feat_means = feat_means.sort_values(by='total', ascending=False)

    return(feat_means['total'])

# Data used in model
data = pd.read_csv(DATA_PATH + 'featurized_data_by_agency.csv')

# Original data from pre-processor
original = pd.read_csv(DATA_PATH + 'clean_data.csv')

X = data.drop(['5_digit_NTD_ID', '5_digit_NTD_ID.1', 'target'], axis=1)
y = data['target']

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=10)

feature_names = X.columns.tolist()

# Tuned LightGBM model (from model.py)

lgb_clf = lgb.LGBMClassifier(n_estimators=650,  # 650
                             boosting_type='gbdt',  # gbdt
                             num_leaves=34,  # 34
                             max_depth=20,  # 20
                             learning_rate=0.01,
                             min_split_gain=0,  # 0
                             min_child_samples=2,  # 2
                             colsample_bytree=0.6,  # 0.6
                             objective='binary',
                             random_state=42,
                             eval_metric='roc_auc',
                             is_unbalance=True,
                             n_jobs=-1)

# lgb_clf = lgb.LGBMClassifier(n_estimators=780,  # 780
#                              boosting_type='gbdt',
#                              num_leaves=13,
#                              max_depth=5,
#                              learning_rate=0.01,
#                              min_split_gain=0,
#                              min_child_samples=2,
#                              colsample_bytree=0.4,
#                              objective='binary',
#                              random_state=42,
#                              eval_metric='roc_auc',
#                              n_jobs=-1)

shuffle_verify(X, y, lgb_clf)

# Permutation Importance

perm = PermutationImportance(lgb_clf, random_state=42).fit(test_X, test_y)

eli5.show_weights(perm, feature_names=feature_names)

# Partial Dependence PLots - outliers make it difficult to see.

def pdp_plotter(feature, model):
    pdp_feat = pdp.pdp_isolate(model=lgb_clf,
                               dataset=test_X,
                               model_features=feature_names,
                               feature=feature)
    pdp.pdp_plot(pdp_feat, feature)
    plt.show()


pdp_plotter('service_to_uza_area', lgb_clf)

# SHAP

# Re-fit the model and extract the SHAP tree explainer Features
# to determine which features fit most often

top_feats = shuffle_SHAP(X, y, lgb_clf, n_shuffles=100)
# top_feats.to_csv(DATA_PATH + 'top_features.csv')

explainer = shap.TreeExplainer(lgb_clf)
shap_values = explainer.shap_values(test_X)
shap.summary_plot(shap_values, test_X)

# SHAP Dependence PLot
shap.dependence_plot("Unlinked_Passenger_Trips_FY", shap_values, test_X)

# Denver RTD [1 - Ridership is Stable / Increasing]
original.loc[original.HQ_City.str.contains('Denver')]
data.loc[data['5_digit_NTD_ID'] == 80006]
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[630,:])
