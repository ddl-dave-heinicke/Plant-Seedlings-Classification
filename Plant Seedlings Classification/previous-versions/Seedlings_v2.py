# Further exploration of the competition data, including masking techniques
# and unsupervised clustering techniques to find natural groupings in the data

# Many thanks to Gábor Vecsei's kernel "Pants PCA & t-SNE" https://www.kaggle.com/gaborvecsei/plants-t-sne

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import cv2
from glob import glob
from sklearn.pipeline import Pipeline, make_pipeline

# sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Data folders
TRAIN_DATA_FOLDER = '.../Seedlings/train'
TEST_DATA_FOLDER = '.../Seedlings/test'

# Definitions

def create_mask_for_plant(image, sensitivity=35):
    # Convert BGR to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of color in HSV
    sensitivity = sensitivity
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    
    # Threshold the HSV image to get only selected colors
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    
    # Create mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image, sensitivity=35):
    mask = create_mask_for_plant(image, sensitivity)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output
    
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
    
def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()
 
# Read and pre-process data
images = []
labels = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (150, 150))
        image = segment_plant(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (45,45))
        
        image = image.flatten()
        
        images.append(image)
        labels.append(class_folder_name)
        
images = np.array(images)
labels = np.array(labels)

original_images = {}

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    plant_name = class_folder_name
    original_images[plant_name] = []
    for image_path in glob(os.path.join(class_folder_path, '*.png')):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #image = cv2.resize(image, (150,150))
        #image = image.flatten()
        original_images[plant_name].append(image)


# Dictionary of image labels 
label_to_id_dict = {v:i for i, v in enumerate(np.unique(labels))}
id_to_label_dict = {v:k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])

# Tune HSV sensitivity

label = id_to_label_dict[6]
class_folder_path = os.path.join(TRAIN_DATA_FOLDER, label)

sensitivities = np.arange(40,51,1)

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.09, wspace=0.09)
fig.suptitle('HSV Sensitivity, {} to {} {}'.format(np.min(sensitivities),np.max(sensitivities),label),  y=1.05, fontsize=14)

for sens, i in zip(sensitivities, range(10)):
    for image_path, j in zip(glob(os.path.join(class_folder_path, "*.png")), range(5)):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (150, 150))
        image = segment_plant(image, sens)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, (45,45))
        image = image.flatten()
        ax = fig.add_subplot(10,5, (i*5 + j+1), xticks=[], yticks=[])
        ax.imshow(np.reshape(image, (150,150)), cmap='Greens')

fig.tight_layout()
plt.show()

# Compare Masks

image = original_images['Small-flowered Cranesbill'][97]

fig, ax = plt.subplots(1, 4, figsize=(20,20))

im_mask = create_mask_for_plant(image)
im_seg = segment_plant(image)
im_sharp = sharpen_image(im_seg)

ax[0].imshow(image)
ax[1].imshow(im_mask)
ax[2].imshow(im_seg)
ax[3].imshow(im_sharp)

plt.show()

# Scale images

images_scaled = StandardScaler().fit_transform(images)

assert images.shape == images_scaled.shape
assert label_ids.shape[0] == images.shape[0]

# PCA

pca = PCA(n_components=180)

pca_result = pca.fit_transform(images_scaled)

pca_result.shape

# t-SNE

tsne = TSNE(learning_rate=100, random_state=42)

tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

tsne_result.shape
tsne_result_scaled.shape
visualize_scatter(tsne_result_scaled, label_ids)


