"""
Machine learning training process.
At the moment developing a neural network for a multi output regression task.
"""
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from Loading_functions import get_custom_loss


# get the model
def get_model(input_shape, sigma):
    '''
    Building the machine learning architecture for model training

    :param input_shape: input shape for the feature data to build the regressor
    :return: tensorflow model
    '''

    tf.random.set_seed(42)

    model = Sequential([

        # normalizer,
        Dense(512, input_shape=(6,), activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(512, activation='LeakyReLU'),
        Dense(45)
    ])

    model.build(input_shape)
    model.summary()

    model.compile(
        loss=get_custom_loss(sigma),
        optimizer=Adam(amsgrad=True, learning_rate=0.005)
    )

    return model


early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,  # how many epochs to wait before stopping
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    'best_model',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Import the training datasets
# feature_file = 'Data/Data_for_ML/training_data/feature'
# label_file = 'Data/Data_for_ML/training_data/label_full_int'

# For subsampling, but if using all 1000 training samples set X_tot and y_tot as X, Y_tot_.
# X = genfromtxt(feature_file)
# y = genfromtxt(label_file)

# print('Feature data shape:', X.shape)
# print('Label data shape: ', y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Save the train and test datasets
# np.save('Data/Data_for_ML/training_data/X_train_900.npy', X_train)
# np.save('Data/Data_for_ML/testing_data/X_test_100.npy', X_test)
# np.save('Data/Data_for_ML/training_data/y_train_900.npy', y_train)
# np.save('Data/Data_for_ML/testing_data/y_test_100.npy', y_test)
# np.save('Data/Data_for_ML/training_data/X_train_900_full_int.npy', X_train)
# np.save('Data/Data_for_ML/testing_data/X_test_100_full_int.npy', X_test)
# np.save('Data/Data_for_ML/training_data/y_train_900_full_int.npy', y_train)
# np.save('Data/Data_for_ML/testing_data/y_test_100_full_int.npy', y_test)

X_train = np.load('Data/Data_for_ML/training_data/X_train_900_full_int.npy')
y_train = np.load('Data/Data_for_ML/training_data/y_train_900_full_int.npy')
frac_sig = np.load('fractional_sigma.npy')

# idx = np.random.choice(np.arange(len(X_train)), 800, replace=False)
# X_train = X_train[idx]
# y_train = y_train[idx]

# Normalize the data to reduce the dynamical range.
# normalizer = preprocessing.Normalization()
# normalizer.adapt(X_train)

print('Feature data shape:', X_train.shape)
print('Label data shape: ', y_train.shape)

input_shape = X_train.shape

# Fit and save models
n_members = 5
for i in range(n_members):

    # Fit model
    model = get_model(input_shape, frac_sig)

    # Log for tensorboard analysis
    model_name = "Ensemble_model_" + str(i + 1) + "_555_mask_900_LRELU_int"
    log_dir = "logs/fit/" + model_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on all data
    start = time.perf_counter()
    history = model.fit(X_train, y_train,
                        verbose=0,
                        validation_split=0.3,
                        callbacks=[early_stopping, tensorboard_callback],
                        epochs=700)

    model.trainable = True

    model.compile(
        optimizer=RMSprop(learning_rate=0.00001),
        loss=masked_mae,
        metrics=[masked_mae]
    )

    start_epoch = len(model.history.history['loss'])

    model.fit(X_train, y_train,
              verbose=0,
              validation_split=0.3,
              callbacks=[early_stopping, tensorboard_callback],
              initial_epoch=start_epoch,
              epochs=3000)

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds' % elapsed, ' for model ' + str(i + 1))

    model.save('Models/' + model_name)
    print('>Saved %s' % model_name)
