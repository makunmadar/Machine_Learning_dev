"""
Machine learning training process.
At the moment developing a neural network for a multi output regression task.
"""
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from Loading_functions import masked_mae


# get the model
def get_model(input_shape):
    '''
    Building the machine learning architecture for model training
    Current configuration maps 11 GALFORM parameters to 102 bins equating to three GALFORM statistics

    :param input_shape: input shape for the feature data to build the regressor
    :return: tensorflow model
    '''

    #tf.random.set_seed(42)

    model = Sequential([

        normalizer,
        Dense(512, input_shape=(11,), activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        # Dense(512, activation='relu'),
        # Dense(512, activation='relu'),
        # Dense(512, activation='LeakyReLU'),

        Dense(102)
    ])

    model.build(input_shape)
    model.summary()

    model.compile(
        loss=masked_mae,
        optimizer=Adam(amsgrad=True, learning_rate=0.005),
        metrics=[masked_mae]
    )
    return model

# Define the callbacks during training.
# Early stopping halts the training once the model no longer an improvement in the validation loss value
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,  # how many epochs to wait before stopping
    restore_best_weights=True,
    verbose=1
)
# Checkpoint saves the epoch with the best validation loss
checkpoint = ModelCheckpoint(
    'best_model',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
################################################
'''
Here we are splitting our total training data into training and testing data. The testing data should be completely
unseen by the network.
Before running the network, please run the code in this section first to save the training and testing data split,
including the exit() statement.

This could probably go in the data_generation script

Import the training datasets: Uncomment everything below including the exit statement for safety.
Load in the relevant data to be split up into training and testing data.
'''

# feature_file = 'Data/Data_for_ML/training_data/feature_2999'
# label_file = 'Data/Data_for_ML/training_data/label_full2999'
#
# # For subsampling, but if using all 1000 training samples set X_tot and y_tot as X, Y_tot_.
# X = genfromtxt(feature_file)
# y = genfromtxt(label_file)
#
# print('Feature data shape:', X.shape)
# print('Label data shape: ', y.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0333, random_state=42)
# print('Testing shape: ', X_test.shape)
# print('Training shape: ', X_train.shape)
#
# # Save the train and test datasets
# np.save('Data/Data_for_ML/training_data/X_train_2899_full.npy', X_train)
# np.save('Data/Data_for_ML/testing_data/X_test_100_full.npy', X_test)
# np.save('Data/Data_for_ML/training_data/y_train_2899_full.npy', y_train)
# np.save('Data/Data_for_ML/testing_data/y_test_100_full.npy', y_test)

# exit()

# Once the training and testing data has been saved, this section can be commented out again.
################################################
X_train = np.load('Data/Data_for_ML/training_data/X_train_2899_full.npy')
y_train = np.load('Data/Data_for_ML/training_data/y_train_2899_full.npy')

# If you want to subsample the training set further use the following code
# idx = np.random.choice(np.arange(len(X_train)), 900, replace=False)
# X_train = X_train[idx]
# y_train = y_train[idx]

# Normalize the data to reduce the dynamical range.
normalizer = preprocessing.Normalization()
normalizer.adapt(X_train)

print('Feature data shape:', X_train.shape)
print('Label data shape: ', y_train.shape)

input_shape = X_train.shape

# We train 5 identical networks. Later on we average over their outputs. This is our ensemble network.

# Fit and save models
n_members = 5
for i in range(n_members):

    # Fit model
    model = get_model(input_shape)

    # Log for tensorboard analysis
    # Save the model as something relevant.
    model_name = "Ensemble_model_" + str(i + 1) + "_6x5_mask_900_LRELU"
    log_dir = "logs/fit/" + model_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on all data
    start = time.perf_counter()  # Monitoring the time taken to train the network
    history = model.fit(X_train, y_train,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[early_stopping, tensorboard_callback],
                        epochs=700)

    model.trainable = True

    # Begin the fine-tuning phase
    model.compile(
        optimizer=RMSprop(learning_rate=0.00001),
        loss=masked_mae,
        metrics=[masked_mae]
    )

    start_epoch = len(model.history.history['loss'])

    model.fit(X_train, y_train,
              verbose=0,
              validation_split=0.2,
              callbacks=[early_stopping, tensorboard_callback],
              initial_epoch=start_epoch,
              epochs=1000)

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds' % elapsed, ' for model ' + str(i + 1))

    # Save the model
    model.save('Models/' + model_name)
    print('>Saved %s' % model_name)
