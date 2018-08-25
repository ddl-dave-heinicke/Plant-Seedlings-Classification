# Final, cleaned-up version before the competition ends!

import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

'''
pic2vec:
Featurize images using a small, contained pre-trained deep learning network
By Jett Oristaglio <jettori88@gmail.com>
https://github.com/datarobot/pic2vec/tree/master/pic2vec

'''
# ImageFeaturizer slightly modified to read through multiple directories
from pic2vec.image_featurizer_multiclass import ImageFeaturizerMulti

'''
ImageFilter created for Seedlings Classification Competition
https://github.com/dheinicke1/auto-adjust-filter/tree/master/auto-adjust-filter
'''
from image_filter import ImageFilter

# Folders to save images
TRAIN_DATA_FOLDER = '.../Seedlings/train'
TEST_DATA_FOLDER = '.../Seedlings/test'
MASK_DATA_FOLDER = '.../Seedlings/train_mask'
MASK_TEST_DATA_FOLDER = '.../Seedlings/test_mask'

#############################################################################

# Definitions

#############################################################################


def kaggle_preds(preds, image_labels, csv_filename):

    kaggle_predictions = pd.DataFrame()

    for pred, filename in zip(preds, image_labels):
        plant = [key for key, value in label_to_id_dict.items()
                 if value == pred]
        plant = plant[0]
        kaggle_predictions = kaggle_predictions.append({'file': filename,
                                                        'species': plant},
                                                       ignore_index=True)

    kaggle_predictions = kaggle_predictions.sort_values(by='file')

    kaggle_predictions.to_csv(csv_filename, index=False)


def plot_conf_matrix(preds, y_test, classes):
    cm = confusion_matrix(preds, y_test)
    abbreviation = [' BG ', ' Ch ', ' Cl ', ' CC ', ' CW ', ' FH ', ' LSB ',
                    ' M ', ' SM ', ' SP ', ' SFC ', ' SB ']
    pd.DataFrame({'class': classes, 'abbreviation': abbreviation})
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, ax=ax, cmap=plt.cm.Greens, annot=True)
    ax.set_xlabel(abbreviation, size=14)
    ax.set_title('Classifier Confusion Matrix')
    ax.legend(classes)
    plt.show()


def ensenble_predictions(proba_1, proba_2, weight_1):

    combined_proba = np.add(proba_1 * weight_1, proba_2 * (1 - weight_1))

    combined_preds = []

    for array in combined_proba:
        combined_preds.append(np.argmax(array))

    combined_preds = np.array(combined_preds)

    return(combined_preds)

#############################################################################

# Load and Pre-process Training Images

#############################################################################

# Use ImageFilter to mask training images


for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = TRAIN_DATA_FOLDER + '/' + class_folder_name

    # See progress
    print(class_folder_name)

    for image_path in glob(os.path.join(class_folder_path, '*.png')):

        # Starting point for finding green plants
        upper_HSV_init = [48, 255, 255]
        lower_HSV_init = [24, 27, 15]

        # Read in the image in color
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Apply the adjustable filter to mask the image
        image_filter = ImageFilter(image=image, upper_HSV=upper_HSV_init,
                                   lower_HSV=lower_HSV_init)
        image_filter.min_shape_size = 150

        processed_image, contours, upper_HSV, lower_HSV =\
            image_filter.find_shapes()

        # Save the masked image to a masked image folder
        save_path = MASK_DATA_FOLDER + '/' + class_folder_name + '/' +\
                                       os.path.basename(image_path) + '.png'

        cv2.imwrite(save_path, processed_image)

# Load and mask Test Images
for image_path in glob(os.path.join(TEST_DATA_FOLDER, '*.png')):

    # Starting point for finding green plants
    upper_HSV_init = [48, 255, 255]
    lower_HSV_init = [24, 27, 15]

    # Read in the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image_filter = ImageFilter(image=image, upper_HSV=upper_HSV_init,
                               lower_HSV=lower_HSV_init)

    image_filter.min_shape_size = 150

    processed_image, contours, upper_HSV, lower_HSV =\
        image_filter.find_shapes()

    # Save the masked image to a masked image folder
    save_path = MASK_TEST_DATA_FOLDER + '/' + os.path.basename(image_path)\
                                            + '.png'
    cv2.imwrite(save_path, processed_image)

# Featurize the masked training images
# Change folder to original images folder to featurize the original images

# Type of pre-trained model to pass to ImageFeaturizerMulti
model = 'xception'

train_images = []

for class_folder_name in os.listdir(MASK_DATA_FOLDER):
    class_folder_path = MASK_DATA_FOLDER + '/' + class_folder_name
    print(class_folder_name)

    featurizer = ImageFeaturizerMulti(depth=2, model=model)

    featurizer.load_data(class_folder_name, image_path=class_folder_path)

    train_images.append(featurizer.featurize())

train_images = np.array(train_images)

# np.save('featurized_xception_2_layer_train_mask.npy', train_images)

# Get training labels
labels = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = MASK_DATA_FOLDER + '/' + class_folder_name
    for image in glob(os.path.join(class_folder_path, '*.png')):
        labels.append(class_folder_name)

labels = np.array(labels)

# Featurize the maksed test images

test_images = []

featurizer = ImageFeaturizerMulti(depth=2, model=model)
featurizer.load_data('test images', image_path=MASK_TEST_DATA_FOLDER)
test_images_mask = (featurizer.featurize())

test_images_mask = np.array(test_images_mask)
# np.save('featurized_xception_2_layer_test_mask.npy', test_images)

###############################################################################
# Alternatively, load previously featurized images as array
# Featurized train images
train_images_original = np.load('featurized_xception_2_layer_train_raw.npy')
train_images_mask = np.load('featurized_xception_2_layer_train_mask_raw.npy')

# Training labels
labels = np.load('labels_original.npy')

# Featurized test images
test_images_original = np.load('featurized_xception_2_layer_test.npy')
test_images_mask = np.load('featurized_xception_2_layer_test_mask.npy')

# Array of test set image filenames, for final submission
test_filenames = np.load('test_filenames.npy')

# Process masked training images array to 2D

X_mask = []

for i in range(train_images_mask.shape[0]):
    for j in range(train_images_mask[i].shape[0]):
        X_mask.append(train_images_mask[i][j])

train_images_mask = np.array(X_mask)

# Process original training images array to 2D

X_ = []

for i in range(train_images_original.shape[0]):
    for j in range(train_images_original[i].shape[0]):
        X_.append(train_images_original[i][j])

train_images_original = np.array(X_)

# Create a dictionary of labels as numeric values, create target y

unique_labels = np.unique(labels)
label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
y = np.array([label_to_id_dict[x] for x in labels])

# Train, test, and split training data

from sklearn.model_selection import train_test_split

# Original
X_train, X_test, y_train, y_test =\
    train_test_split(train_images_original, y, test_size=0.2, random_state=15)

# Masked
X_train_mask, X_test_mask, y_train_mask, y_test_mask =\
    train_test_split(train_images_mask, y, test_size=0.2, random_state=15)

#############################################################################

# Fit featurizedd data to various models

#############################################################################


# Logistic regression
#############################################################################

# Train #

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

logreg = LogisticRegression(penalty='l2', tol=0.001, C=0.15,
                            multi_class='multinomial', solver='lbfgs',
                            max_iter=300)

logreg.fit(X_train, y_train)

original_preds_logreg = logreg.predict(X_test)
original_preds_logreg_proba = logreg.predict_proba(X_test)

print('Score original images: %f' % (logreg.score(X_test, y_test)))

logreg.fit(X_train_mask, y_train_mask)

mask_preds_logreg = logreg.predict(X_test_mask)
mask_preds_logreg_proba = logreg.predict_proba(X_test_mask)

print('Score masked images: %f' % (logreg.score(X_test_mask, y_test_mask)))

# Create confusion matrix of each to comapre
plot_conf_matrix(original_preds_logreg, y_test, classes=unique_labels)

plot_conf_matrix(mask_preds_logreg, y_test, classes=unique_labels)

# Ensenble masked and unmasked image models
mask_weight = 0.57

combined_preds = ensenble_predictions(mask_preds_logreg_proba,
                                      original_preds_logreg_proba,
                                      mask_weight)

print(accuracy_score(combined_preds, y_test))

plot_conf_matrix(combined_preds, y_test, unique_labels)

# Optimize mask weight

weights = np.linspace(0, 1.0, 20)  # ~0.57

for weight in weights:
    combined_pred = ensenble_predictions(mask_preds_logreg_proba,
                                         original_preds_logreg_proba,
                                         weight)

    score = accuracy_score(combined_pred, y_test)
    print('Weight: %f \t Score: %f' % (weight, score))

# Test #

test_mask_pred_proba = logreg.predict_proba(test_images_mask)

test_original_pred_proba = logreg.predict_proba(test_images_original)

combined_preds_test = ensenble_predictions(test_mask_pred_proba,
                                           test_original_pred_proba,
                                           0.57)

kaggle_preds(combined_preds_test, test_filenames,
             'LogisticRegressionEnsenble_preds.csv')


# XGBoost #
#############################################################################

import xgboost as xgb

# Classifier parameters found previously using RandomSearchCV

xgb_clf = xgb.XGBClassifier(max_depth=200,  # 50
                            learning_rate=0.1,
                            n_estimators=750,
                            objective='multi:softmax',
                            gamma=0,
                            subsample=1)

# Original images ~ 84% Accurate
xgb_clf.fit(X=X_train, y=y_train)
original_preds_xgb = xgb_clf.predict(X_test)
original_preds_xgb_proba = xgb_clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, original_preds_xgb)
print(accuracy)

# Masked Images ~ 85.8% Accurate
xgb_clf.fit(X=X_train_mask, y=y_train_mask)
mask_preds_xgb = xgb_clf.predict(X_test_mask)
mask_preds_xgb_proba = xgb_clf.predict_proba(X_test_mask)
accuracy = accuracy_score(y_test_mask, mask_preds_xgb)
print(accuracy)


# K Nearest Neighbors ~79.6% accurate (original)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train, y_train)

knn_preds = knn.predict(X_test)
print(knn.score(X_test, y_test))

plot_conf_matrix(knn_preds, y_test, unique_labels)

# Neural Net model with fully connected layers
#############################################################################

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

num_classes = len(unique_labels)
epochs = 100
batch_size = 64

y_train_OHE = to_categorical(y_train)
y_test_OHE = to_categorical(y_test)

y_train_OHE_mask = to_categorical(y_train_mask)
y_test_OHE_mask = to_categorical(y_test_mask)


model = Sequential()
model.add(Dense(1024, activation='sigmoid', input_shape=X_train.shape[1:]))
model.add(Dropout(0.4))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=5)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

history_original = model.fit(X_train,
                             y_train_OHE,
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[early_stopping_monitor],
                             validation_data=(X_test, y_test_OHE),
                             verbose=2)

original_preds_keras_proba = model.predict(X_test, batch_size=batch_size,
                                           verbose=2)

original_preds_keras = original_preds_keras_proba.argmax(axis=1)

print(accuracy_score(original_preds_keras, y_test))

history_mask = model.fit(X_train_mask,
                         y_train_OHE_mask,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=[early_stopping_monitor],
                         validation_data=(X_test_mask, y_test_OHE_mask),
                         verbose=2)

mask_preds_keras_proba = model.predict(X_test_mask, batch_size=batch_size,
                                       verbose=2)

mask_preds_keras = mask_preds_keras_proba.argmax(axis=1)

print(accuracy_score(mask_preds_keras, y_test))

plot_conf_matrix(original_preds_keras, y_test, classes=unique_labels)

mask_keras_preds = model.predict(test_images_mask).argmax(axis=1)

kaggle_preds(mask_keras_preds, test_filenames, 'Keras_mask_preds.csv')

#############################################################################

# Ensenble model outputs

#############################################################################

# Optimize masked weight - Neural Net & XGBoost

weights = np.linspace(0, 1.0, 20)  # ~0.42, 88.7%

for weight in weights:
    combined_pred = ensenble_predictions(mask_preds_xgb_proba,
                                         mask_preds_keras_proba, weight)
    score = accuracy_score(combined_pred, y_test_mask)
    print('Weight: %f \t Score: %f' % (weight, score))

# Optimize original weight - Neural Net & XGBoost

weights = np.linspace(0, 1.0, 20)  # ~0.36, 90.4%

for weight in weights:
    combined_pred = ensenble_predictions(original_preds_xgb_proba,
                                         original_preds_keras_proba, weight)
    score = accuracy_score(combined_pred, y_test)
    print('Weight: %f \t Score: %f' % (weight, score))

# Optimize masked weight - Neural Net & LogisticRegression

weights = np.linspace(0, 1.0, 20)  # 1, 0.63, 91.6%

for weight in weights:
    combined_pred = ensenble_predictions(mask_preds_logreg_proba,
                                         mask_preds_keras_proba, weight)
    score = accuracy_score(combined_pred, y_test_mask)
    print('Weight: %f \t Score: %f' % (weight, score))

# Optimize original weight - Neural Net & LogisticRegression

weights = np.linspace(0, 1.0, 20)  # ~0.58, 91%

for weight in weights:
    combined_pred = ensenble_predictions(original_preds_logreg_proba,
                                         original_preds_keras_proba, weight)
    score = accuracy_score(combined_pred, y_test)
    print('Weight: %f \t Score: %f' % (weight, score))

# Logreg & Keras Ensenble Preditions #
#############################################################################


# Combination of Logistic Regression and Neural Net performs the best.

# Neural net predictions (probabilities)
test_original_preds_keras_proba = model.predict(test_images_original,
                                                batch_size=batch_size)
test_mask_preds_keras_proba = model.predict(test_images_mask,
                                            batch_size=batch_size)

# Neural net predictions
test_original_preds_keras = test_original_preds_keras_proba.argmax(axis=1)

# Logisitc Regression predictions (probabilities)
test_original_preds_logreg_proba = logreg.predict_proba(test_images_original)
test_mask_preds_logreg_proba = logreg.predict_proba(test_images_mask)

# Combine neural net and logisitc regression predictions
# Unmasked images
combined_pred_original = ensenble_predictions(test_original_preds_logreg_proba,
                                              test_original_preds_keras_proba,
                                              0.57)
# Masked Images
combined_pred_mask = ensenble_predictions(test_mask_preds_logreg_proba,
                                          test_mask_preds_keras_proba,
                                          0.65)

# Saveand submit predictions
# The featurized original images do a little better on the unseen data, despite
# higher scores on the validation holdout set. Could be some overfitting to the
# training images.

kaggle_preds(combined_pred_original, test_filenames,
             'Logreg_keras_original.csv')

kaggle_preds(combined_pred_mask, test_filenames, 'Logreg_keras_mask.csv')
