"""
Machine learning training process.
Testing a Functional API network.
"""
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import time
import random
from Loading_functions import masked_mae


# Creation of model
def create_model3():
    input1 = tf.keras.Input(shape=(6,), name='I1')
    input2 = tf.keras.Input(shape=(6,), name='I2')

    hidden1 = Dense(units=512, activation='LeakyReLU')(input1)
    hidden2 = Dense(units=512, activation='LeakyReLU')(input2)
    hidden3 = Dense(units=512, activation='LeakyReLU')(hidden1)
    hidden4 = Dense(units=2, activation='LeakyReLU')(hidden2)
    output1 = Dense(units=49, name='O1')(hidden3)
    output2 = Dense(units=18, name='O2')(hidden4)

    model = tf.keras.models.Model(inputs=[input1, input2], outputs=[output1, output2])

    model.summary()

    model.compile(optimizer=Adam(amsgrad=True, learning_rate=0.005),
                  loss=masked_mae,
                  metrics=[masked_mae])

    return model

# Load in the training data
X_train = np.load('Data/Data_for_ML/training_data/X_train_900_full.npy')
y_train = np.load('Data/Data_for_ML/training_data/y_train_900_full.npy')
yz_train = np.array([i[0:49] for i in y_train])
yk_train = np.array([i[49:67] for i in y_train])

print('Feature data shape:', X_train.shape)
print('Label data shape: ', y_train.shape)
print('z label data shape: ', yz_train.shape)
print('k label data shape: ', yk_train.shape)

model = create_model3()

# tf.keras.utils.plot_model(model, 'FAPI.png', show_shapes=True)
# Log for tensorboard analysis
model_name = "API_model_555_mask_900_RELU"
log_dir = "logs/fit/" + model_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    x={'I1': X_train, 'I2': X_train},
    y={'O1': yz_train, 'O2': yk_train},
    epochs=200,
    verbose=0,
    validation_split=0.3,
    callbacks=[tensorboard_callback]
)

# Testing the model
# X_test = np.load('Data/Data_for_ML/testing_data/X_test_100_full.npy')
# y_test = np.load('Data/Data_for_ML/testing_data/y_test_100_full.npy')
# yz_test = np.array([i[0:49] for i in y_test])
# yk_test = np.array([i[49:67] for i in y_test])
# bin_file = 'Data/Data_for_ML/bin_data/bin_full'
# bins = genfromtxt(bin_file)
#
# model.save('Models/' + model_name)
