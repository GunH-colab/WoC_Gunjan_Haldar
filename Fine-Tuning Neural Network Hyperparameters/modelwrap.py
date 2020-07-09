'''
P Y T H O N   F I L E   D E S C R I P T I O N:

 this is a Simple Sequential model for univariate regression using an SGD opitimizer with the specified learning rate 
 with the default 
 -> input shapes, 
 -> the number of layers,
 -> only one output neuron
 
# creating a baseline functiom that will build and compile a keras regression model.

'''

# MAKE YOU CHANGES HERE (IF YOU WANT TO)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

hidden_layers = 1 # number of hidden layers set to default as 1. You may chage it according to your model
neuron_numbers = 30 # number of neurons per layer set to default as 30. You may chage it according to your model
rate_of_learning = 3e-3 # learing rate set to default. You may chage it according to your model

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

from tensorflow import keras

def model_for_tuning(hiddenN = hidden_layers, neuronsN = neuron_numbers, learnR = rate_of_learning, input_shape = [13]):
    # input_shape is [13] to fit the bostonhousing dataset as i am using that for example. You may chage it according to the model you want to build

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layers in range(hiddenN):
        model.add(keras.layers.Dense(neuronsN, activation="relu"))
    model.add(keras.layers.Dense(1))

    optim = keras.optimizers.SGD(lr = learnR)
    model.compile(loss="mse", optimizer = optim, metrics = ["mape"])

    return model


'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''