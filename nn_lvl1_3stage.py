import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import cv2


def importData():

    # can be automated for multiple levels
    stage1 = np.load("level1_1.npy")
    stage2_1 = np.load("level1_2_1.npy")
    stage2_2 = np.load("level1_2_2.npy")
    stage3 = np.load("level1_3.npy")

    data = []

    for img in stage1:
        data.append([img, 0])

    for img in stage2_1:
        data.append([img, 1])

    for img in stage2_2:
        data.append([img, 1])

    for img in stage3:
        data.append([img, 2])

    return data


def processData(data):

    shuffle(data)

    X = []
    y = []

    for img, label in data:
        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # scale the data
    X = X/255.0

    return X, y


def trainData(X, y):
    
    # build the model
    model = tf.keras.models.Sequential()
    # flatten the input layer
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu)) # relu is the default activation function

    # output layer
    model.add(tf.keras.layers.Dense(3, activation = tf.nn.softmax)) #softmax is used for probability distribution

    # parameters to train the model
    model.compile(optimizer = 'adam', # Stochastic gradient descent
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
    # to train the model
    model.fit(X, y, epochs = 10, validation_split = 0.3)

    return model


# MAIN RUN
data = importData()

X, y = processData(data)

model = trainData(X, y)

# Saving

model.save('lvl1_3stages')
