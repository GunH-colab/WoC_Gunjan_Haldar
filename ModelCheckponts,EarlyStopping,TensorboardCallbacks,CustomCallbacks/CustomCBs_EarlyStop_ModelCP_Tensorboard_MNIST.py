'''
P Y T H O N   F I L E   D E S C R I P T I O N :
MNIST Dataset is used !!!!!!!!

-> An example to set Callbacks while or after training your model
-> Using TensorBoard for visualisation

# Callbacks used:

-> ModelCheckpoint (saves checkpoints of the model in regular intervals)
-> TensorBoard (for visualisation)
-> EarlyStopping(stops training tf there is no change in train/val loss)
-> a custom CallBack(to print the ration b/w validation and train loss)

NOTE !!!!! 

-> generate_subdir_tensorboard.py should be in the same directory as this(its provided together with this python file)
-> creating_custom_callbacks.py should be in the same directory as this(its provided together with this python file)

'''

#importing all the necessary libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# my own modules
import generate_subdir_tensorboard as gst
import creating_custom_callbacks as ccc

print(tf.__version__)
print(keras.__version__)

# fetching the MNIST dataset from keras

digits = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = digits.load_data()

# normalizing the dataset(by dividing it with 255.0) and further creating a validation set by taking a slice(5000) from train set

valid_images, train_images = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
valid_labels, train_labels = y_train_full[:5000], y_train_full[5000:]
test_images = X_test/255.0
test_labels = y_test

#class_names array 

class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# viewing the images(tracing pixel density)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.show()

# rechecking the fetched dataset on grayscale

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#viewing the dimensions of every segment of the dataset

print(train_images.shape,"\n", train_labels.shape,"\n", test_images.shape,"\n", test_labels.shape,"\n", valid_images.shape,"\n", valid_labels.shape,"\n")

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# compiling the model

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


#   C A L L B A C K S     S E C T I O N

run_log_dir = gst.create_dir_cur()

check = keras.callbacks.ModelCheckpoint("my_model3_cbs.h5", save_best_only=True)
ES = keras.callbacks.EarlyStopping(monitor = 'val_loss', verbose=1, mode = 'min', patience=10, restore_best_weights=True)
tensor_check = keras.callbacks.TensorBoard(run_log_dir)

history = model.fit(train_images, train_labels, epochs=50, validation_data=(valid_images, valid_labels), callbacks=[check, ES, ccc.CustomCallbacks(), tensor_check])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) 

#testing the model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''