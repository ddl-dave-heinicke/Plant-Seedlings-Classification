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


# DATA_PATH = 'C:\\Users\\Dave\\Documents\\Python Scripts\\Transit\\'
DATA_PATH = 'C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Transit\\'

data = pd.read_csv(DATA_PATH + 'featurized_data_by_agency.csv')

X = data.drop(['5_digit_NTD_ID', 'target'], axis=1)
y = data['target']

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=2)

 feature_names = X.columns.tolist()

# LightGBM model

lgb_clf = lgb.LGBMClassifier(n_estimators=650,
                             boosting_type='gbdt',
                             num_leaves=14,
                             max_depth=5,
                             learning_rate=0.01,
                             min_split_gain=0,
                             min_child_samples=3,
                             colsample_bytree=1,
                             objective='binary',
                             random_state=42,
                             eval_metric='roc_auc',
                             n_jobs=-1)

model = lgb_clf.fit(train_X, train_y)
preds = model.predict(test_X)
roc_auc_score(test_y, preds)

# Permutation Importance

perm = PermutationImportance(model, random_state=42).fit(test_X, test_y)

eli5.show_weights(perm, feature_names=feature_names)

# Partial Dependence PLots

def pdp_plotter(feature, model):
    pdp_feat = pdp.pdp_isolate(model=model,
                               dataset=test_X,
                               model_features=feature_names,
                               feature=feature)
    pdp.pdp_plot(pdp_feat, feature)
    plt.show()


pdp_plotter('cost_per_mile', model)
