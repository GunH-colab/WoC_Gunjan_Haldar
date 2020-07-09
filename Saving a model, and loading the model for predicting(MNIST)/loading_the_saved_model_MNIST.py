'''
P Y T H O N   F I L E   D E S C R I P T I O N :

An example To call a saved model and use it for prediction

NOTE !!!!!

The model(h5 file) saved should in the same directory where you run the code. In this case its 'my_model.h5'
'''

#impporting all the necessary libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(tf.__version__)
print(keras.__version__)

# fetching the MNIST dataset from keras

digits = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = digits.load_data()

# class_names array

class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# normalizing the dataset(by dividing it with 255.0) and further creating a validation set by taking a slice(5000) from train set

valid_images, train_images = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
valid_labels, train_labels = y_train_full[:5000], y_train_full[5000:]
test_images = X_test/255.0
test_labels = y_test

model = keras.models.load_model("my_model.h5")

predictions = model.predict(test_images)

#function to view the image and with labels(predicted class, score%, actual class)

def plotImg(i, preds_arr, true_label, img):
    true_label, img = true_label[i], img[i]

    plt.yticks([])
    plt.xticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    pred_label = np.argmax(preds_arr)

    if pred_label == true_label:
        COLOR = 'green'
    else:
        COLOR = 'red'

    plt.xlabel("{} {:0.2f}% ({})".format(class_names[pred_label], 100*np.max(preds_arr), class_names[true_label]), color=COLOR)

#function to construct bar graphs of the scores of each class

def plotGraph(i, preds_arr, true_label):
    true_label = true_label[i]

    plt.xticks(range(10))
    plt.yticks([])
    plt.ylim([0, 1])
    PLOT = plt.bar(range(10), preds_arr, color="#777777")

# for a successfull prediction color is 'green' else 'red'
    PLOT[np.argmax(preds_arr)].set_color('red')
    PLOT[true_label].set_color('green')

# to produce a mass tight sheet of predictions and their graph

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plotImg(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plotGraph(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''