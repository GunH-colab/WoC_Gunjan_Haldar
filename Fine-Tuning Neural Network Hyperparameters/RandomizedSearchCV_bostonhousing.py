'''
P Y T H O N   F I L E   D E S C R I P T I O N :
Bostonhousing Dataset is used !!!!!!!!

This file will simply try to fit the model in as many combinations of hyperparameters in a particular range provided by us. We wil try to explore the hyperparameter
space using RandomizedSearchCV module. We will need to wrap our keras baseline regressor model first using KerasRegressor

NOTE !!!!! 

-> modelwrap.py should be in the same directory as this(its provided with this python file)
-> the model is modelwrap.py is structured to work on bostonhousing dataset. To work on other datasets you might have to rechange the 'input_shape' 

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
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# defining our hyperparameter space
params_set = {
    "hiddenN": [0, 1, 2 , 3, 4, 5, 6, 7, 8, 9],
    "neuronsN": np.arange(1, 1000),
    "learnR": reciprocal(3e-4, 3e-2)
}

# wrapping

reg_keras = keras.wrappers.scikit_learn.KerasRegressor(mw.model_for_tuning)

# using the 'best hyperparameters finder' module

rsCV = RandomizedSearchCV(reg_keras, params_set, n_iter=10, cv=3)

# Its on !!!

rsCV.fit(train_data_std, train_targets, epochs=1000, validation_split = 0.2, callbacks=[keras.callbacks.EarlyStopping(patience = 100)])

print(rsCV.best_params_,"\n\n\n", rsCV.best_score_)

model = rsCV.best_estimator_.model

# now you can save the model or check its performance

print("\n", model.evaluate(test_data_std, test_targets))

'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''