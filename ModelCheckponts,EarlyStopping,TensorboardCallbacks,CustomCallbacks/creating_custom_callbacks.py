# an example # to display the ratio b/w val_loss and train_loss

from tensorflow import keras

# syntax:
class CustomCallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs):
        print("\nValidation:Train {:.2f}".format(logs["val_loss"]/logs["loss"]))
'''
# SYNTAX:
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
import creating_custom_callbacks as ccc
history = model.fit(................. callbacks=[ccc.CustomCallbacks()])
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
 


# Other examples of custom CallBacks:


    while training(called by fit())

    # on_train_begin(self, logs):
    # on_train_end(self, logs)
    # on_epoch_begin(self, epoch, logs)
    # on_epoch_end(self, epoch, logs)
    # on_train_batch_begin(self, batch, logs)
    # on_train_batch_end(self, batch, logs)

    while evaluating(called by evaluate())

    # on_test_begin(self, logs)
    # on_test_end(self, logs)
    # on_test_batch_begin(self, batch, logs)
    # on_test_batch_end(self, batch, logs)

    while predicting(called by predict())

    # on_predict_begin(self, logs)
    # on_predict_end(self, logs)
    # on_predict_batch_begin(self, batch, logs)
    # on_predict_batch_end(self, batch, logs)
'''

'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''

