# Begin developing an image filter that automatically filters the image
# Inspired by the image sliders created using CV2

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
from sklearn.cross_validation import train_test_split

# Data folders
TRAIN_DATA_FOLDER = '.../Seedlings/train'
TEST_DATA_FOLDER = '.../Seedlings/test'

# Default Filters

lower_HSV = np.array([35, 100, 5]) 
upper_HSV = np.array([200, 255, 255])

# Definitions

# HSV Definitions
def HSV_mask(image, lower_HSV, upper_HSV):
    mask = cv2.inRange(image, lower_HSV, upper_HSV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
    
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
 
def process_image(image, lower_HSV=lower_HSV, upper_HSV=upper_HSV):
    image = sharpen_image(image)
    mask = HSV_mask(image, lower_HSV, upper_HSV)
    output = cv2.bitwise_and(image, image, mask = mask)
    (_, contours, heiarchy) = cv2.findContours(mask, cv2.RETR_TREE, 
    cv2.CHAIN_APPROX_SIMPLE)
    return output, contours, heiarchy

# Sliders for adjusting HSV on sample image 
def LowerH(val, lower_HSV=lower_HSV):
    global sample_image
    lower_HSV[0] = val
    image, contours, heiarchy = process_image(sample_image, lower_HSV)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    cv2.imshow('Filtered Image', image)
    
def LowerS(val, lower_HSV=lower_HSV):
    global sample_image
    lower_HSV[1] = val
    image, contours, heiarchy = process_image(sample_image, lower_HSV)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    cv2.imshow('Filtered Image', image)
    
def LowerV(val, lower_HSV=lower_HSV):
    global sample_image
    lower_HSV[2] = val
    image, contours, heiarchy = process_image(sample_image, lower_HSV)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    cv2.imshow('Filtered Image', image)
    
# Summary of countours found for each species in training set

def contour_summary(labels, filenames, n_contours): 
    df_contours = pd.DataFrame()
    df_contours = pd.DataFrame({'label':labels,'filename':filenames, 'n_contours':n_contours})
    df_contours = df_contours.set_index(['label','filename'])
    df_contours = df_contours.sort_index()
    
    df_summary = pd.DataFrame(index=np.unique(labels), columns=['min num contours','max num contours','frac_with_contours'])
    
    for label in np.unique(labels):
        desc = df_contours.loc[(label)].describe()
        df_summary.loc[(label)]['min num contours'] = desc.loc[('min','n_contours')]
        df_summary.loc[(label)]['max num contours'] = desc.loc[('max','n_contours')]
        df_summary.loc[(label)]['frac_with_contours'] = np.count_nonzero(df_contours.loc[(label)]) / desc.loc[('count','n_contours')]
    
    print(df_summary)

# Main

# Read and pre-process data
images_fixed = []
labels_fixed = []
filenames_fixed = []
n_contours_fixed = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image, contours, heiarchy = process_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (150,150))
        
        image = image.flatten()
        
        images_fixed.append(image)
        labels_fixed.append(class_folder_name)
        filenames_fixed.append(os.path.basename(image_path))
        n_contours_fixed.append(len(contours))
        
images_fixed = np.array(images_fixed) 
labels_fixed = np.array(labels_fixed)
filenames_fixed = np.array(filenames_fixed)
n_contours_fixed = np.array(n_contours_fixed)

contour_summary(labels_fixed, filenames_fixed, n_contours_fixed)

# Adjust HSV values on sample image

lower_HSV = np.array([31, 100, 10]) 
upper_HSV = np.array([84, 255, 255])

sample_image = cv2.imread('0bdaf1d8f.png', cv2.IMREAD_COLOR)
sample_image = cv2.resize(sample_image, (300, 300))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
processed_image, contours, heiarchy = process_image(sample_image)
print(count_contours(contours, 1500))

cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
cv2.drawContours(processed_image, contours, -1, (0, 0, 255), 1)
cv2.imshow("Filtered Image", processed_image)

cv2.createTrackbar('Hue', 'Filtered Image', lower_HSV[0], upper_HSV[0], LowerH)
cv2.createTrackbar('Saturation', 'Filtered Image', lower_HSV[1], upper_HSV[1], LowerS)
cv2.createTrackbar('Value', 'Filtered Image', lower_HSV[2], upper_HSV[2], LowerV)

cv2.waitKey(0)
cv2.destroyAllWindows()

# View samples

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.09, wspace=0.09)
#fig.suptitle('Lower HSV Params, Hue: {} Saturation: {} Value: {}'.format(lower_HSV[0], lower_HSV[1], lower_HSV[2]),  y=1.05, fontsize=14)

for class_folder_name, i in zip(os.listdir(TRAIN_DATA_FOLDER), range(12)):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    for image_path, j in zip(glob(os.path.join(class_folder_path, "*.png")), range(5)):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image, contours, heiarchy = process_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        ax = fig.add_subplot(12,5, (i*5 + j+1), xticks=[], yticks=[])
        ax.imshow(np.reshape(image, (200,200)), cmap='Greens')
        #print("Species: %s sample %d : %d" % (class_folder_name, j, len(contours)))

fig.tight_layout()
plt.show()

################################## 
#Test adjust filter on single sample image

def create_mask(image, upper_HSV, lower_HSV):
    mask = cv2.inRange(image, lower_HSV, upper_HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    (_, contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return masked_image, contours
    
def count_contours(contours, min_shape_size):
    num = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_shape_size:
            num = num + 1
    return(num)

def tune_hue(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower):
    
    contour_counter = 0

    while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
        masked_image, contours = create_mask(image, upper_HSV, lower_HSV)
        contour_counter = count_contours(contours, min_shape_size)
        
        # 1
        lower_HSV[0] = lower_HSV[0] - 2
        upper_HSV[0] = upper_HSV[0] + 2
        
        #print('Hue', lower_HSV[0])
        
        if lower_HSV[0] < 20:
            break
        
        # if lower_HSV[0] >= upper_HSV[0]:
        #     break
    
    return(masked_image, contours, upper_HSV, lower_HSV)
        
def tune_sat(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower):
    
    contour_counter = 0
    init_hue_upper = upper_HSV[0]
    init_hue_lower = lower_HSV[0]
    
    while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
        
        processed_image, contours, upper_HSV, lower_HSV = tune_hue(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower)
        
        contour_counter = count_contours(contours, min_shape_size)
        
        if contour_counter >= contour_threshold_lower and contour_counter <= contour_threshold_upper:
            break
        
        # 1
        lower_HSV[1] = lower_HSV[1] - 5
        
        upper_HSV[0] = init_hue_upper
        lower_HSV[0] = init_hue_lower
        
        #print('SAT:' , lower_HSV[1])
        
        if lower_HSV[1] <= 45:
            break
    
    return(processed_image, contours, upper_HSV, lower_HSV)    

def find_shapes(image, upper_HSV, lower_HSV, contour_threshold_upper = 3, contour_threshold_lower = 1):
    
    contour_counter = 0
    min_shape_size = 3500
      
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = sharpen_image(image)
    
    init_sat_upper = upper_HSV[1]
    init_sat_lower = lower_HSV[1]
    
    while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
        
        processed_image, contours, upper_HSV, lower_HSV = tune_sat(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower)
        
        contour_counter = count_contours(contours, min_shape_size)
        
        # 100
        min_shape_size = min_shape_size -200
        #print(min_shape_size)
        
        if contour_counter >= contour_threshold_lower and contour_counter <= contour_threshold_upper:
            break
        
        upper_HSV[1] = init_sat_upper
        lower_HSV[1] = init_sat_lower
        
        if min_shape_size < 50:
            print('No contours found')
            break
    
    return(processed_image, contours, upper_HSV, lower_HSV)

def find_the_plant(image):
    
    # Initial filter
    upper_HSV = np.array([50, 255, 255])
    lower_HSV = np.array([35, 50, 10]) 
    contour_threshold_upper = 6
    contour_threshold_lower = 1
    
    processed_image, contours, upper_HSV, lower_HSV = find_shapes(image, upper_HSV, lower_HSV, contour_threshold_upper, contour_threshold_lower)
    
    return(processed_image, contours, upper_HSV, lower_HSV)

# Main

images = []
labels = []
filenames = []
n_contours = []

for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        processed_image, contours, upper_HSV, lower_HSV = find_the_plant(image)
        processed_image = cv2.resize(processed_image, (150,150))
        processed_image = processed_image.flatten()
        images.append(processed_image)
        labels.append(class_folder_name)
        filenames.append(os.path.basename(image_path))
        n_contours.append(len(contours))
        
images = np.array(images) 
labels = np.array(labels)
filenames = np.array(filenames)
n_contours = np.array(n_contours)
unique_labels = np.unique(labels)
unique_labels
contour_summary(labels, filenames, n_contours)

#np.save('processed_images.npy', images)
# np.save('labels.npy', labels)
# np.save('filenames.npy', filenames)
# np.save('n_contours.npy', n_contours)


# View samples

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.09, wspace=0.09)
#fig.suptitle('Lower HSV Params, Hue: {} Saturation: {} Value: {}'.format(lower_HSV[0], lower_HSV[1], lower_HSV[2]),  y=1.05, fontsize=14)

skip = 15

for class_folder_name, i in zip(os.listdir(TRAIN_DATA_FOLDER), range(12)):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    print(class_folder_name)
    for image_path, j in zip(glob(os.path.join(class_folder_path, "*.png")), range(5+skip)):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if j > skip:
            image, contours, upper_HSV, lower_HSV = find_the_plant(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            ax = fig.add_subplot(12,5, (i*5 + j+1-skip), xticks=[], yticks=[])
            ax.imshow(np.reshape(image, (300,300)), cmap='Greens')

fig.tight_layout()
plt.show()

# Adjust Single Image


sample_image = cv2.imread('0bdaf1d8f.png', cv2.IMREAD_COLOR)
# sample_image = cv2.resize(sample_image, (300, 300))
# sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
image, contours, upper_HSV, lower_HSV = find_the_plant(sample_image)
#print(count_contours(contours, 1500))
print('Lower HSV', lower_HSV)
cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
cv2.drawContours(processed_image, contours, -1, (0, 0, 255), 1)
cv2.imshow("Filtered Image", processed_image)

# cv2.createTrackbar('Hue', 'Filtered Image', lower_HSV[0], upper_HSV[0], LowerH)
# cv2.createTrackbar('Saturation', 'Filtered Image', lower_HSV[1], upper_HSV[1], LowerS)
# cv2.createTrackbar('Value', 'Filtered Image', lower_HSV[2], upper_HSV[2], LowerV)

cv2.waitKey(0)
cv2.destroyAllWindows()

