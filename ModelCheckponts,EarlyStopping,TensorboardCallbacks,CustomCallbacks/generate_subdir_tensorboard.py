'''
P Y T H O N   F I L E   D E S C R I P T I O N :

# FOR TENSORBOARD callback and visualising it
-> This will set a directory for the Tensorboard files named 'TensorBoard_Logs'
-> Below function that will create a separate sub directory for every run describing the current time and date of the run
-> This will create a sub directory for every run

'''

import os
import time

root_dir = os.path.join(os.curdir, "TensorBoard_Logs")

def create_dir_cur():
    import time
    run_time = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_dir, run_time)

'''
# SYNTAX:
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
import directory_for_every_run as dfer

run_log_dir = dfer.create_dir_cur()

tensor_check = keras.callbacks.TensorBoard(run_log_dir)

history = model.fit(..............., callbacks=[............., tensor_check]
----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# this will create the file necessary for TensorBoard visualisations........


# Go to the folder containing 'Tensorboard_Logs', open terminal and type:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
>>>tensorboard --logdir=./Tensorboard_logs --port=6006
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Open a web browser and visit the link
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
http://localhost:6006/
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''



'''
==================================================================================================================================================================
---signed by 'GunH-Colab'
'''


