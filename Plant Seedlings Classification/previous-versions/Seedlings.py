# Initial exploration of the competition data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import skimage
from skimage import data, transform
from skimage.color import rgb2gray

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".png")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(str(d))
    return images, labels

# Load Images    
# Data Folders
ROOT_PATH = '.../Seedlings/'

train_data_directory = os.path.join(ROOT_PATH, "train")
test_data_directory = os.path.join(ROOT_PATH, "test")

images_s, labels_s = load_data(train_data_directory)

images_s = np.array(images_s)
labels_s = np.array(labels_s)

# View a sample image
print(images_s.ndim)
print(images_s.size)
print(images_s[0])

print(labels_s.ndim)
print(labels_s.size)
print(labels_s[0])
print(len(set(labels_s)))

# Inspect distribution of images

plt.hist(labels_s, bins=len(set(labels_s)))
plt.xticks(rotation = 'vertical')
plt.show()

# Display samples of each type of seedling

labels_s_tags = np.unique(labels_s)
rows = labels_s_tags.tolist()

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.09, wspace=0.09)
fig.suptitle('Sample Images', y=1.05, fontsize=14)

for i in range(len(labels_s_tags)):
    current = images_s[labels_s == labels_s_tags[i]]
    for j in range(12):
        ax = fig.add_subplot(12,12, (i*12 + j+1), xticks=[], yticks=[])
        ax.imshow(current[j], interpolation='nearest')
        if j == 0:
            ax.set_ylabel(rows[i], rotation=0, size='large', ha='right')

images_300 = [transform.resize(image, (300,300), mode='constant') for image in images_s]
images_300[0].shape
images_300 = np.array(images_300)
images_300.shape

# Verify image dimenstions
images_2D = []

for i in range(len(images_300)):
    if len(images_300[i].shape) != 3:
        images_2D.append(image)

images_gray = rgb2gray(images_300)


fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.09, wspace=0.09)
fig.suptitle('Sample Images', y=1.05, fontsize=14)

for i in range(len(labels_s_tags)):
    current = images_gray[labels_s == labels_s_tags[i]]
    for j in range(12):
        ax = fig.add_subplot(12,12, (i*12 + j+1), xticks=[], yticks=[])
        ax.imshow(current[j], interpolation='nearest')
        if j == 0:
            ax.set_ylabel(rows[i], rotation=0, size='large', ha='right')
            
            
