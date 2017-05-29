"""
    Personal mini-project with Deep Learning and Keras
    Identify emails by their authors. Authors and labels:
    Sara has label 0
    Chris has label 1
"""

# Datasets:  https://github.com/udacity/ud120-projects/tree/master/tools

import sys
from time import time

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

labels_train = keras.utils.to_categorical(labels_train,2)
labels_test = keras.utils.to_categorical(labels_test,2)


features_train = np.array(features_train)

print(features_test.shape)

# Building the model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim = features_train.shape[1]))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(2, activation='softmax'))

#Activation function: relu and sigmoid
#Loss function: categorical_crossentropy, mean_squared_error
#Optimizer: rmsprop, adam, ada

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

t0 = time()
model.fit(features_train, labels_train, epochs=5, batch_size=256, verbose=2)
t1 = time()

# pred = model.predict(features_test)

accuracy = model.evaluate(features_test, labels_test, verbose=2)
t2 = time()

print("accuracy: \t\t {:.6f}".format(accuracy[1]))
print("train time(s): \t {:.6f}".format(t1-t0))
print("test time(s): \t {:.6f}".format(t2-t2))