# Work
os.chdir('C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Seedlings')
# Home
#os.chdir('C:\\Users\\Dave\\Google Drive\\Data Science Training\\Python Scripts\\Seedlings')


import os
import pandas as pd
import numpy as np
import cv2
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

#from pic2vec import ImageFeaturizer
#from pic2vec.image_featurizer_multiclass import  ImageFeaturizerMulti
from image_filter import ImageFilter

# Sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import TruncatedSVD, PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler, scale, MaxAbsScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# from sklearn.tree import  DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier

# xgboost
#import xgboost as xgb

# Home
# TEMP_TRAIN_DATA_FOLDER = 'C:/Users/Dave/Python Scripts/Pics'
# TRAIN_DATA_FOLDER = 'C:/Users/Dave/Documents/Python Scripts/Seedlings/train'
# TEST_DATA_FOLDER = 'C:/Users/Dave/Documents/Python Scripts/Seedlings/test'

# Work
TRAIN_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/train_aug'
TEST_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/test'
MASK_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/train_aug_masked'

#TEMP = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/temp'

# Work
#TRAIN_DATA_FOLDER = 'C:/Users/dheinicke/Google Drive/Data Science Training/Python Scripts/Seedlings/train'
#TEST_DATA_FOLDER = 'C:/Users/dheinicke/Google Drive/Data Science Training/Python Scripts/Seedlings/test'

############################################################################# 

# Main

# Pre-process Images

masked_images = []
labels = []
filenames = []
n_contours = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = TRAIN_DATA_FOLDER + '/' +  class_folder_name + '/output'
    #class_folder_path = os.path.join(TEMP, class_folder_name)
    print(class_folder_name)
    
    for image_path in glob(os.path.join(class_folder_path, '*.JPEG')):
        
        upper_HSV_init = np.array([50,255,255])
        lower_HSV_init = np.array([35, 50, 10])
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image_filter = ImageFilter(image=image, upper_HSV=upper_HSV_init, lower_HSV=lower_HSV_init)
        processed_image, contours, upper_HSV, lower_HSV = image_filter.find_shapes()
        
        save_path = MASK_DATA_FOLDER + '/' + class_folder_name + '/' + os.path.basename(image_path) + '.JPEG'
        cv2.imwrite(save_path, processed_image)
        
        masked_images.append(processed_image)
        labels.append(class_folder_name)
        filenames.append(os.path.basename(image_path))
        n_contours.append(len(contours))
      
masked_images = np.array(masked_images)
labels = np.array(labels)
filenames = np.array(filenames)
n_contours = np.array(n_contours)

# 2 Featurize Images
## Append the contours feature??

model = 'xception'
df_featurized = pd.DataFrame()
x = [] 
labels = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = TRAIN_DATA_FOLDER + '/' +  class_folder_name + '/output'
    print(class_folder_name)
    
    #featurizer = ImageFeaturizerMulti(depth=2, model=model)
    
    #featurizer.load_data(class_folder_name, image_path=class_folder_path)
    
    #x.append(featurizer.featurize())
    
    for image_path in glob(os.path.join(class_folder_path, "*.JPEG")):
        labels.append(class_folder_name)


#x = np.array(x)
#X = []
#X_2 = []
#X_2_aug = []


# for i in range(x.shape[0]):
#     for j in range(x[i].shape[0]):
#         X_2_aug.append(x[i][j])

#X = np.load('featurized_xception_1_layer.npy')        
labels = np.array(labels)
y = pd.get_dummies(labels)

#np.save('featurized_xception_2_layer_aug.npy', X_2_aug)

X = np.load('featurized_xception_2_layer_aug.npy')

print(X[0].shape)
print(len(X))

unique_labels = np.unique(labels)
label_to_id_dict = {v:i for i, v in enumerate(np.unique(labels))}
label_ids = np.array([label_to_id_dict[x] for x in labels])

# PCA
# pca = PCA(n_components=150, random_state=42)

# transformed = pca.fit_transform(X)

# lr =100
# tsne = TSNE(n_components=2, learning_rate=lr)

# tsne_features = tsne.fit_transform(transformed)

# tsne_df = pd.DataFrame(tsne_features, columns = ['x','y'])
# tsne_df['labels'] = labels

# sns.lmplot(x='x', y='y', data=tsne_df, fit_reg=False, hue='labels', markers='labels', legend=True)

# Decision tree classifier

# X = StandardScaler().fit_transform(X)

# vals = [375, 400, 425, 450, 475, 500]

# #for max_features in vals:
# dt_clf = DecisionTreeClassifier(max_depth=150, max_features=400)
# dt_clf.fit(X_train, y_train)
# preds = dt_clf.predict(X_test)
# score = accuracy_score(y_test, preds)
# print('Depth: %i Score: %f' %(400, score))

# # Random Forrest Classifier

# RFC = RandomForestClassifier(n_estimators=150, max_features=400, max_depth=150)
# RFC.fit(X_train, y_train)
# preds = RFC.predict(X_test)
# score = accuracy_score(y_test, preds)
# print(score)

# XGBoost
X_scaled = MaxAbsScaler().fit_transform(X)

dmatrix = xgb.DMatrix(data=X, label=label_ids)
num_boosting_round = 30

params = {'objective':'multi:softmax', 
          'eta': 0.2, 
          'gamma': 0, 
          'max_depth': 50,
          'num_class':12, 
          'learning_rate': 0.1, 
          'n_estimators': 750,
          'subsample': 1,
         }

xgb_cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=3, num_boost_round=num_boosting_round, metrics="merror", seed=42)
print(xgb_cv_results)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.arange(1,num_boosting_round + 1,1), xgb_cv_results['test-merror-mean'])
ax.set_xticks(np.arange(1,num_boosting_round+ 1,1))
plt.show()

# Score 0.8156 max_depth = 100, lr = 0.1, n_estimators = 400
    
# xgb_param_grid = {'max_depth' : [100], #100
#                   'learning_rate': [0.1], #0.1
#                   'n_estimators' : [400,500,1000], #400
#                   }

# xgb_clf = xgb.XGBClassifier(objective = 'multi:softmax')

# grid_search = GridSearchCV(estimator=xgb_clf,
#                                       param_grid=xgb_param_grid,
#                                       n_jobs = -1,
#                                       scoring = 'accuracy',
#                                       cv=3,
#                                       verbose=1)

# grid_search.fit(X,label_ids)

# print(grid_search.best_params_)
# print(grid_search.best_score_)

#
#max_depth=xgb_param_grid['max_depth'], 
#                             learning_rate=xgb_param_grid['learning_rate'], 
#                             n_estimators=xgb_param_grid['n_estimators'],
#                             objective='multi:softmax',
#                             n_jobs= -1,
#                             subsample = 0.9)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')

X_train, X_test, y_train, y_test = train_test_split(X, label_ids, test_size=0.1, random_state=42)
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)
score = logreg.score(X_test, y_test)


# Confusion Matrix
cm = confusion_matrix(preds, y_test)
abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
pd.DataFrame({'class': unique_labels, 'abbreviation': abbreviation})
fig, ax = plt.subplots(1)
ax = sns.heatmap(cm, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xlabel(abbreviation)
ax.set_ylabel(abbreviation)
plt.show()

# SGD

from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = train_test_split(X, label_ids, test_size=0.1, random_state=42)
    
alphas = [0.0025, 0.005, 0.0075, 0.01] # 0.005

for alpha in alphas:
    sgd = SGDClassifier(alpha=alpha, shuffle=True)
    sgd.fit(X_train, y_train)
    preds = sgd.predict(X_test)
    score = sgd.score(X_test, y_test)
    print('Alpha: %f score %f' %(alpha, score))


# Dense neural net model

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping

# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# X_scaled = StandardScaler().fit_transform(X)
# X_scaled = np.array(X_scaled)
# X = np.array(X)
# y = np.array(y)

# input_shape = (X_scaled.shape[1],)

# model = Sequential()
# model.add(Dense(1024, activation='tanh', input_shape = input_shape )) 
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu')) 
# model.add(Dropout(0.5))
# model.add(Dense(12, activation='softmax')) 
# model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# early_stopping_monitor = EarlyStopping(patience=3)

# hist = model.fit(X, y, 
#           epochs=10, 
#           batch_size=64, 
#           #callbacks=[early_stopping_monitor], 
#           validation_split= 0.1,
#           verbose=2)


