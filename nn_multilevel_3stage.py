import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import cv2


def importData():

    # can be automated for multiple levels
    files = ["level1_1.npy", "level1_2_1.npy", "level1_2_2.npy", "level1_3.npy",
            "level2_1.npy", "level2_2_1.npy", "level2_2_2.npy", "level2_3_1.npy", "level2_3_2.npy",
            "level3_1_1.npy", "level3_1_2.npy", "level3_2_1.npy", "level3_2_2.npy", "level3_3_1.npy", "level3_3_2.npy"]
    files = list(map((lambda x: "training/" + x), files))

    data = []

    for f in files:
        stage = np.load(f)

        for img in stage:
            data.append([img, int(f[16]) - 1])

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

model.save('multilevel_3stages')
