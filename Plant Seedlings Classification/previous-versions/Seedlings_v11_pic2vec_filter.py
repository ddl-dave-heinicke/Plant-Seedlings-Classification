# Work
#os.chdir('C:\\Users\\dheinicke\\Google Drive\\Data Science Training\\Python Scripts\\Seedlings')
# Home
os.chdir('C:\\Users\\Dave\\Google Drive\\Data Science Training\\Python Scripts\\Seedlings')


import os
import pandas as pd
import numpy as np
import cv2
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

from pic2vec import ImageFeaturizer
from pic2vec.image_featurizer_multiclass import  ImageFeaturizerMulti
from image_filter import ImageFilter

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, scale, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# xgboost
import xgboost as xgb

# Home
#TEMP_TRAIN_DATA_FOLDER = 'C:/Users/Dave/Python Scripts/Pics'
TRAIN_DATA_FOLDER = 'C:/Users/Dave/Documents/Python Scripts/Seedlings/train'
TEST_DATA_FOLDER = 'C:/Users/Dave/Documents/Python Scripts/Seedlings/test'

# Work
# TRAIN_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/train_aug'
# #TRAIN_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/train'
# TEST_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/test_jpeg'
# MASK_DATA_FOLDER = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/train_aug_masked'

#TEMP = 'C:/Users/dheinicke/Documents/Python Scripts/Seedlings/temp'

# Work
#TRAIN_DATA_FOLDER = 'C:/Users/dheinicke/Google Drive/Data Science Training/Python Scripts/Seedlings/train'
#TEST_DATA_FOLDER = 'C:/Users/dheinicke/Google Drive/Data Science Training/Python Scripts/Seedlings/test'

############################################################################# 

# Load and Pre-process Images

masked_images = []
labels = []
filenames = []
n_contours = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = TRAIN_DATA_FOLDER + '/' +  class_folder_name
    #class_folder_path = os.path.join(TEMP, class_folder_name)
    print(class_folder_name)
    
    for image_path in glob(os.path.join(class_folder_path, '*.png')):
        
        # upper_HSV_init = np.array([50,255,255])
        # lower_HSV_init = np.array([35, 50, 10])
        
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # image_filter = ImageFilter(image=image, upper_HSV=upper_HSV_init, lower_HSV=lower_HSV_init)
        # processed_image, contours, upper_HSV, lower_HSV = image_filter.find_shapes()
        
        # save_path = MASK_DATA_FOLDER + '/' + class_folder_name + '/' + os.path.basename(image_path) + '.JPEG'
        # cv2.imwrite(save_path, processed_image)
        
        # masked_images.append(processed_image)
        labels.append(class_folder_name)
        #filenames.append(os.path.basename(image_path))
        #n_contours.append(len(contours))
      
masked_images = np.array(masked_images)
labels = np.array(labels)
filenames = np.array(filenames)
n_contours = np.array(n_contours)

# 2 Featurize Images
## Append the contours feature??

model = 'xception'
labels = []
x = [] 

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = TRAIN_DATA_FOLDER + '/' +  class_folder_name
    print(class_folder_name)
    
    featurizer = ImageFeaturizerMulti(depth=2, model=model)
    
    featurizer.load_data(class_folder_name, image_path=class_folder_path)
    
    x.append(featurizer.featurize())
    
x = np.array(x)

np.save('featurized_xception_2_layer_train_raw.npy', x)

#X = []
#X_2 = []
#X_2_aug = []
#X_2_aug_filter = []

for i in range(x.shape[0]):
    for j in range(x[i].shape[0]):
        X_2_aug.append(x[i][j])

#X_2_aug = np.array(X_2_aug)

# test_filenames = []
# #X_test_images = []

# for image_path in glob(os.path.join('C:/Users/dheinicke/Documents/Python Scripts/Seedlings/test', "*.png")):
#     test_filenames.append(os.path.basename(image_path))

# test_filenames = np.array(test_filenames)

# Featurize test images

featurizer = ImageFeaturizerMulti(depth=2, model=model)
featurizer.load_data('test images', image_path=TEST_DATA_FOLDER)
test_images = (featurizer.featurize())
test_images = np.array(test_images)

test_labels = pd.read_csv('test_labels_featurizer.csv')
test_labels = test_labels.as_matrix()
test_labels = np.array(test_labels,dtype='str').flatten()

featurizer.load_data('test', image_path='C:/Users/Dave/Documents/Python Scripts/Seedlings')
sample = featurizer.featurize()
sample = np.array(sample)

#np.save('featurized_xception_2_layer_test.npy', test_images)
#np.save('featurized_xception_2_layer_aug.npy', X_2_aug)
#np.save('featurized_xception_2_layer_aug_filtered.npy', X_2_aug_filter)
#np.save('labels.npy', labels)
#np.save('original_labels.npy', labels)
#np.save('featurized_xception_2_layer_test_JPEG.npy', X_test_images)
#np.save('test_filenames.npy', test_filenames)
#np.save('test_filenames.npy', test_labels)
#labels = np.load('original_labels.npy')
X = np.load('featurized_xception_2_layer_aug_raw.npy')
labels = np.load('labels_aug.npy')

X_ = []
X.shape

for i in range(X.shape[0]):
    for j in range(X[i].shape[0]):
        X_.append(X[i][j])

X_ = np.array(X_)

X_.shape

test_images = np.load('featurized_xception_2_layer_test.npy')
#test_filenames = np.load('test_filenames.npy')
test_images.shape
#filenames = np.load('filenames.npy')
#y = pd.get_dummies(labels)

unique_labels = np.unique(labels)
label_to_id_dict = {v:i for i, v in enumerate(np.unique(labels))}
label_ids = np.array([label_to_id_dict[x] for x in labels])

X_shuffle, label_ids_shuffle = shuffle(X_, label_ids, random_state=42)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_, label_ids, test_size=0.2, random_state=1)

logreg = LogisticRegression(penalty='l2', tol = 0.0001, C= 0.15, multi_class='multinomial', solver='lbfgs', max_iter=300)
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)

score = accuracy_score(y_test, preds)
print(score)

test_preds = logreg.predict(test_images)

kaggle_predictions =pd.DataFrame()

for pred, filename in zip(test_preds, test_labels):
    plant = [key for key, value in label_to_id_dict.items() if value == pred]
    plant = plant[0]
    kaggle_predictions = kaggle_predictions.append({'file': filename, 'species':plant}, ignore_index=True)

kaggle_predictions = kaggle_predictions.sort_values(by='file')
kaggle_predictions.head(10)

kaggle_predictions.to_csv('LogisticRegression_preds_aug.csv', index=False)

for label in np.unique(kaggle_predictions.species):
    a = sum(kaggle_predictions.species == label)
    print('%s %i' %(label, a))

# Confusion Matrix
cm = confusion_matrix(preds, y_test)
abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
pd.DataFrame({'class': unique_labels, 'abbreviation': abbreviation})
fig, ax = plt.subplots(1)
ax = sns.heatmap(cm, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xlabel(abbreviation)
ax.set_ylabel(abbreviation)
ax.set_title('Logistic Regression Classifier Confusion Matrix')
plt.show()

# Ridge Regression

from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
alphas = [0.00001, 0.0005, 0.001]

ridge = RidgeClassifier(alpha=0.01)
ridge.fit(X_train,y_train)
preds = ridge.predict(X_test)
score = ridge.score(X_test, y_test)

print(score)

test_preds = ridge.predict(X_test_images)

kaggle_predictions =pd.DataFrame()

for pred, filename in zip(test_preds, test_labels):
    plant = [key for key, value in label_to_id_dict.items() if value == pred]
    plant = plant[0]
    kaggle_predictions = kaggle_predictions.append({'file': filename, 'species':plant}, ignore_index=True)

kaggle_predictions = kaggle_predictions.sort_values(by='file')
kaggle_predictions.head(10)

kaggle_predictions.to_csv('RidgeRegression_preds.csv', index=False)



# XGBoost 
# Filtered 82.8% at 200 boosting rounds
# Unfiltered 88%

X_scaled = MaxAbsScaler().fit_transform(X)

dmatrix = xgb.DMatrix(data=X_, label=label_ids)
num_boosting_round = 100

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

xgb_clf = xgb.XGBClassifier(max_depth=50, 
                            learning_rate=0.1, 
                            n_estimators=750, 
                            objective='multi:softmax', 
                            gamma=0, 
                            subsample=1)

dmatrix = xgb.DMatrix(data=X_train, label=y_train)

X_train, X_test, y_train, y_test = train_test_split(X, label_ids, test_size=0.1, random_state=42)

xgb_clf.fit(X=X_train, y=y_train)
preds = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print(accuracy)

xgb_test_preds = xgb_clf.predict(X_test_images)

kaggle_predictions =pd.DataFrame()

for pred, filename in zip(xgb_test_preds, test_filenames):
    plant = [key for key, value in label_to_id_dict.items() if value == pred]
    plant = plant[0]
    kaggle_predictions = kaggle_predictions.append({'file': filename, 'species':plant}, ignore_index=True)

kaggle_predictions.head()
kaggle_predictions.to_csv('XGB_preds.csv', index=False)


# Confusion Matrix
cm = confusion_matrix(preds, y_test)
abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
pd.DataFrame({'class': unique_labels, 'abbreviation': abbreviation})
fig, ax = plt.subplots(1)
ax = sns.heatmap(cm, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xlabel(abbreviation)
ax.set_ylabel(abbreviation)
ax.set_title('XGBoost Classifier Confusion Matrix')
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


# SGD - 86.9%

from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = train_test_split(X_filter, label_ids, test_size=0.1, random_state=42)
    
alphas = [0.0025, 0.005, 0.0075, 0.01] # 0.005

for alpha in alphas:
    sgd = SGDClassifier(alpha=alpha, shuffle=True)
    sgd.fit(X_train, y_train)
    preds = sgd.predict(X_test)
    score = sgd.score(X_test, y_test)
    print('Alpha: %f score %f' %(alpha, score))

# Naive Bayes -  54%

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
score = gnb.score(X_test, y_test)
print(score)

# Decision tree classifier
# dt_clf = DecisionTreeClassifier(max_depth=150, max_features=400)

# # Random Forrest Classifier
# RFC = RandomForestClassifier(n_estimators=150, max_features=400, max_depth=150)


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

# PCA
# pca = PCA(n_components=150, random_state=42)

# transformed = pca.fit_transform(X)

# lr =100
# tsne = TSNE(n_components=2, learning_rate=lr)

# tsne_features = tsne.fit_transform(transformed)

# tsne_df = pd.DataFrame(tsne_features, columns = ['x','y'])
# tsne_df['labels'] = labels

# sns.lmplot(x='x', y='y', data=tsne_df, fit_reg=False, hue='labels', markers='labels', legend=True)

