"""
Machine learning training process.
At the moment developing a neural network for a multi output regression task.
"""

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

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
        optimizer=Adam(),
        metrics=[rmse, 'accuracy']
    )
    return model


# Evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # Define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Enumerate folds
    for train_ix, test_ix in cv.split(X):
        # Prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # Define model
        model = get_model(n_inputs, n_outputs)
        # Fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # Evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=0)
        # Store result
        print('>%.3f' % mae)
        results.append(mae)
    return results


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1.25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Redshift Distribution]')
    plt.legend()
    plt.grid(True)
    plt.show()


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
history = model.fit(X, y, verbose=0, epochs=150,
                    validation_split=0.2)
plot_loss(history)

model.save('Models/my_model')

# Evaluate model
# results = evaluate_model(X, y)
# Summarize performance
# print('MAE: %.3f (%.3f)' % (mean(results), std(results)))
