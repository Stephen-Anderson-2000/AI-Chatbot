# -*- coding: utf-8 -*-
"""
Created on Sat Jan 2 10:51:39 2021

@author: Stephen
"""

## Use this python file separately to train and save the CNN model
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential

## Following 2 lines are used to show available GPU computing power
#import tensorflow as tf
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#dataDirectory = "../Datasets/Vehicle/Classes"
#dataDirectory = "../Datasets/Small-Vehicle/Classes"
dataDirectory = "../Datasets/Vehicle/"
filePath = "Required-Files/"
trainingSeed = 357


# Batch is set to 32 to improve efficiency during training
batch = 32
# Uses keras to simplify the data and reduce the colours in the images
normalizationLayer = layers.experimental.preprocessing.Rescaling(1./255)


def Get_Training_Data():
    # Loads the training images into a dataframe
    trainData = preprocessing.\
        image_dataset_from_directory(dataDirectory, labels = "inferred",
                                     validation_split = 0.2,
                                     subset = "training", seed = trainingSeed,
                                     batch_size = batch)
    numClasses = len(trainData.class_names)
    normalizedTrain = trainData.map(lambda x, y: (normalizationLayer(x), y))
    return normalizedTrain, numClasses


def Get_Validation_Data():
    # Loads the validation images into a dataframe
    valData = preprocessing.\
        image_dataset_from_directory(dataDirectory, labels = "inferred", 
                                     validation_split = 0.2,
                                     subset = "validation", 
                                     seed = trainingSeed,
                                     batch_size = batch)
    normalizedVal = valData.map(lambda x, y: (normalizationLayer(x), y))
    return normalizedVal


def Make_Model(numClasses):
    # Provides some randomisation to the dataset to help prevent overfitting
    augmentationLayer = Sequential([
            layers.experimental.preprocessing.\
                RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomZoom(0.1)
            ])
    
    model = Sequential([
        augmentationLayer,
        layers.Conv2D(16, 3, padding = "same", activation = "relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding = "same", activation = "relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding = "same", activation = "relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding = "same", activation = "relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation = "relu"),
        layers.Dense(numClasses),
        layers.Dropout(rate = 0.25),
        layers.Dense(numClasses, activation="softmax")
        ])
        
    return model

def Train_Model(model, trainData, valData, numEpochs):
    model.compile(optimizer = "adam",
              loss = keras.losses.\
                  SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])
    
    # Use CUDA GPU compute power to significantly reduce execution time
    model.fit(trainData, validation_data = valData, epochs = numEpochs)
    
    model.summary()
        
def Save_Model(model, fileName): 
    fileName += ".h5"
    model.save(filePath + fileName)

def Load_Model(fileName): # Function used to verify that the model saved okay
    loadedModel = keras.models.load_model(filePath + fileName)
    return loadedModel

def Display_Training_Results(history, numEpochs, fileName):
    acc = history.history["accuracy"]
    valAcc = history.history["val_accuracy"]
    
    loss = history.history["loss"]
    valLoss = history.history["val_loss"]
    
    epochsRange = range(numEpochs)
    
    plt.figure(figsize = (8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochsRange, acc, label = "Training Accuracy")
    plt.plot(epochsRange, valAcc, label = "Validation Accuracy")
    # Graph increases so this keeps the key out of the way
    plt.legend(loc = "lower right")
    plt.title("Training and Validation Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(epochsRange, loss, label = "Training Loss")
    plt.plot(epochsRange, valLoss, label = "Validation Loss")
    # Graph decreases so this keeps the key out of the way
    plt.legend(loc = "upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(fileName)
    plt.show()
    

trainData, numClasses = Get_Training_Data()
valData = Get_Validation_Data()

# Takes a long time per epoch for diminishing returns past 80 epochs
#numEpochs = 100
numEpochs = 5
model = Make_Model(numClasses)
#model = Load_Model(fileName)
Train_Model(model, trainData, valData, numEpochs)

#fileName = "CNN-Model_256-Dense_Vehicles-Dataset_100-Epochs_0-1-Zoom"
fileName = "CNN-Demo"
Save_Model(model, fileName)
Display_Training_Results(model.history, numEpochs, fileName)
