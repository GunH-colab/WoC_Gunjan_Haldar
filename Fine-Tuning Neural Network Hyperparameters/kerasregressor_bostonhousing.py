'''
P Y T H O N   F I L E   D E S C R I P T I O N :
Bostonhousing Dataset is used !!!!!!!!

To import a baseline model and use it as a KerasRegressor...

NOTE !!!!! 

-> modelwrap.py should be in the same directory as this(its provided together with this python file)

'''

#importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import modelwrap as mw

# splitting the datasets into train and test sets

(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()

print(train_data.shape,"\n", test_data.shape,"\n", train_targets.shape,"\n", test_targets.shape)

# pre-processing the datasets using StandardScaler

sc = StandardScaler()
sc.fit(train_data)
train_data_std = sc.transform(train_data)
test_data_std = sc.transform(test_data)

# KerasRegressor object is like an envelope which surrounds the keras model wrapper so that we can use our model like a normal scikit-learn KerasRegressor

reg_keras = keras.wrappers.scikit_learn.KerasRegressor(mw.model_for_tuning)

# traing the model
reg_keras.fit(train_data_std, train_targets, epochs=1000, validation_split = 0.2, callbacks=[keras.callbacks.EarlyStopping(patience = 100)])

# scores :-)

test_score = reg_keras.score(test_data_std, test_targets)
print(test_score)

'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''