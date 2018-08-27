# Plant Seedlings Classification Challenge

[Kaggle Playground Plant Seedling Classification Challenge](https://www.kaggle.com/c/plant-seedlings-classification)

[Data Here](https://www.kaggle.com/c/plant-seedlings-classification/data)

The code for the various models are in the ["Previous Versions"](https://github.com/dheinicke1/Sample-Work/tree/master/Plant%20Seedlings%20Classification/previous-versions) folder, the code used for my final submission is 
["Plant_Seedlings_Classification.py"](https://github.com/dheinicke1/Sample-Work/blob/master/Plant%20Seedlings%20Classification/Plant_Seedlings_Classification.py)

## Objective

The goal of the competition is to build an image recognition model that can differentiate crops from weeds.

Using a training dataset of 4,750 labeled color images of seedlings, correctly classify 
the images as one of 12 different species. 

The model is scored using the prediction accuracy on ~800 unlabeled images. No prize money, just a fun 
exploration of image featurization techniques, image processing and application for pre-trained neural
network models!

## Methodology

This is a classification problem. There are 12 classes of species that can only be classified using image data. 
There is no additional information about the images provided (such as age of the seedling, type of camera used etc.)

How good should the model be? The goal is to recognize crops and weeds, so the consequences of a false positive or 
false negative aren't terrible. 90% accuracy among 12 classes would be a reasonable goal - sufficiently useful in 
practice, without wasting too much time attempting a perfect score.

To tackle the problem, I took the three steps:

**1) Enhance Data** 

There are image pre-processing techniques that could improve model performance can filter images to remove noise or irrelevant data from the image, or extract features such a number of contours, image contours, edges etc. 

Here is an image pre-processor called ["Find the Plant"](https://github.com/dheinicke1/auto-adjust-filter) I created to filter the seedling images. 

We could also artificially increase the size of the training set using data enhancement. Randomly rotating, zooming or cropping images can simulate a larger training set to reduce the risk of the model over-fitting to quirky images in the training set.

**2) Train A Simple Neural Net**

I don't have access to a GPU, and am training my models on my home laptop. This limits the complexity of a model that I can train, but maybe a simple model to specialize in the training images. I reduced the image sizes to 150 X 150 pixels and converted them to grayscale. 

In the first version [(Version 5)](https://github.com/dheinicke1/Sample-Work/blob/master/Plant%20Seedlings%20Classification/previous-versions/Seedlings_v5.1.py) I trained the model and saved the model weights, in later versions imported the pretrained model weights and
experimented with modifying the top layers to improve performance.

Here is the base sequential model architecture:

        # Base Model
        # Conv2D & LeakyReLU with max pooling and dropout layer. Repeated twice.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(150,150,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(150,150,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(0.20))
        
        model.add(Conv2D(64, (3,3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(64, (3,3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(0.20))
        
        # Similar block, ending with higher dropout rate
        model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same')) 
        model.add(Dropout(0.3))
        
        # End of Base Model 
        
        # Top layers. These are re-trained in later versions
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))

The simple, fully-trained model did alright, achieving about 85% classification accuracy on the test set.
Pretty good, but not quite to the goal of 90%.

Keeping the images in color or full scale could improve performance, but my machine could barely handle
the simple model! Time to save some time by investigating pre-trained models.

**3) Pre-Trained Models**

Pre-trained image recognition models are designed to classify images into a number of pre-defined image
classes (dogs, cars, furniture). The models work by identifying patterns in the images, then patterns of
patterns, and then patterns of those patterns until a unique image signature is determined. 

The cool thing about pre-trained models is that the initial layers break down all images into general features, and 
its only the very top layers that actually classify the image. So, a pre-trained model can be generalized to
classify a specific group of images (say types of plant seedlings) by simply allowing the top few layer to be 
re-trained.

Even cooler (to me anyway), the top layer of the models can be removed or "decapitated," and a layer of feature weights
(which are just numeric interpretations of the image patterns) can be fed into *any machine learning model,* such as a 
liner regression model, decision tree or boosting algorithm. And these model outputs can be ensembled, creating an
even more robust predictive model.

Ultimately, I got to my goal of exceeding 90% classification accuracy by feeding the pre-trained features into 
Logistic Regression and simple fully-connected neural net models, and combining those results.

Whew. Mission accomplished.


