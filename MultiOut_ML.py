"""
Machine learning training process.
At the moment developing a neural network for a multi output regression task.
"""

from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import time

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
        Dense(512, input_shape=(6,), activation='sigmoid'),
        Dense(512, activation='sigmoid'),
        Dense(24)
    ])

    model.build(input_shape)
    model.summary()

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=Adam(amsgrad=True, learning_rate=0.005),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model


early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30, # how many epochs to wait before stopping
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
feature_file = 'Data/Data_for_ML/training_data/feature'
label_file = 'Data/Data_for_ML/training_data/label_sub12_dndz'

X = genfromtxt(feature_file)
y = genfromtxt(label_file)

# Half sample test
# X = X[1::2]
# y = y[1::2]

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

# Fit and save models
n_members=5
for i in range(n_members):
    # Fit model
    model = get_model(input_shape)

    # Log for tensorboard analysis
    model_name = "Ensemble_model_"+ str(i+1)
    log_dir = "logs/fit/" + model_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on all data
    start = time.perf_counter()
    history = model.fit(X, y,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[early_stopping, tensorboard_callback],
                        epochs=700)

    model.trainable = True

    model.compile(
        optimizer=Adam(amsgrad=True, learning_rate=0.00001),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    model.fit(X, y,
              verbose=0,
              validation_split=0.2,
              callbacks=[early_stopping, tensorboard_callback],
              epochs=700)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds' % elapsed, ' for model %d' % i+str(1))

    start = time.perf_counter()
    model.save('Models/'+model_name)
    print('>Saved %s' % model_name)

