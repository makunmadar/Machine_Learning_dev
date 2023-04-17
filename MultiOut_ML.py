"""
Machine learning training process.
At the moment developing a neural network for a multi output regression task.
"""

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import math

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# get the model
def get_model(input_shape):
    '''
    Building the machine learning architecture for model training

    :param input_shape: input shape for the feature data to build the regressor
    :return: tensorflow model
    '''

    tf.random.set_seed(42)

    model = Sequential([

        # Currently using Ed's emulator architecture
        Dense(124, input_shape=(6,), activation='sigmoid'),
        Dense(124, activation='sigmoid'),
        Dense(6)
    ])

    model.build(input_shape)
    model.summary()

    model.compile(
        loss=rmse,
        optimizer=Adam(learning_rate = 0.01),
        metrics=[rmse]
    )
    return model


def plot_loss(history, label):
    #plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label=label)
    plt.ylim([0.5, 1.25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Redshift Distribution]')
    plt.legend()
    plt.grid(True)
    #plt.show()

initial_learning_rate = 0.01
epochs = 150
decay = initial_learning_rate/ epochs
def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate*math.pow(drop_rate, math.floor(epoch/epochs_drop))

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)


# Import the training datasets
feature_file = 'Data/Data_for_ML/training_data/feature'
label_file = 'Data/Data_for_ML/training_data/label_sub6_dndz'

X = genfromtxt(feature_file)
y = genfromtxt(label_file)

# Half sample test
#X = X[1::2]
#y = y[1::2]

# Normalize the data to reduce the dynamical range.
# This uses a minmaxscalar where a minimum and maximum are specified.
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X)
X = scaler_feat.transform(X)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y)
y = scaler_label.transform(y)

print('Feature data shape:', X.shape)
print('Label data shape: ', y.shape)

input_shape = X.shape

# Get model
model = get_model(input_shape)

# Fit the model on all data
history = model.fit(X, y,
                    verbose=0, epochs=epochs,
                    validation_split=0.2)
plot_loss(history, 'Fixed LR: 0.01')
history = model.fit(X, y,
                    verbose=0, epochs=epochs,
                    validation_split=0.2,
                    callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=0)])
plot_loss(history, 'Time based lr')
history = model.fit(X, y,
                    verbose=0, epochs=epochs,
                    validation_split=0.2,
                    callbacks=[LearningRateScheduler(lr_step_decay, verbose=0)])
plot_loss(history, 'Step based lr')
history = model.fit(X, y,
                    verbose=0, epochs=epochs,
                    validation_split=0.2,
                    callbacks=[LearningRateScheduler(lr_exp_decay, verbose=0)])
plot_loss(history, 'Exp based lr')

plt.show()
#model.save('Models/my_model')

