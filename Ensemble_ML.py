from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint


early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30, # how many epochs to wait before stopping
    restore_best_weights=True,
    verbose=1
)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_all_models(n_models):
    """
    Load models from file

    :param n_models: total number of tensorflow models
    :return: list of model members
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/model_Ensemble_model_'+str(i+1)
        model = tf.keras.models.load_model(filename, custom_objects={'rmse': rmse}, compile=False)
        # Add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models

def define_stacked_model(members):
    """
    Define stacked model from multiple member input models

    :param members: List of model members
    :return: An integrated ensemble model
    """

    # Update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # Make non trainable
            layer.trainable = False
            # Rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_'+str(i+1)+'_'+layer.name
            # Define multi-headed input
            ensemble_visible = [model.input for model in members]
            # Concatenate merge output from each model
            ensemble_outputs = [model.output for model in members]
            merge = concatenate(ensemble_outputs)
            hidden = Dense(124, activation='sigmoid', name="1")(merge)
            output = Dense(6)(hidden)
            model = Model(inputs=ensemble_visible, outputs=output)
            # Plot graph of ensemble
            #plot_model(model, show_shapes=True, to_file='model_graph.png')
            # Compile
            model.compile(loss=rmse, optimizer=Adam(amsgrad=True, learning_rate=0.005), metrics=[rmse])
    return model

def fit_stacked_model(model, inputX, inputy):
    """
    Fit a stacked model

    :param model: Ensemble model
    :param inputX: Size of input X
    :param inputy: Size of input y
    :return: Fitted model
    """

    # Prepare input data
    X = [inputX for _ in range(len(model.input))]
    # Encode output data
    log_dir = "logs/fit/stacked_model"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Fit model
    model.fit(X, inputy, validation_split=0.2, epochs=700, verbose=0, callbacks=[early_stopping, tensorboard_callback])

# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub6_dndz'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test = scaler_label.transform(y_test)

# Load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Define ensembled model
stacked_model = define_stacked_model(members)
# Fit stacked model on test dataset
fit_stacked_model(stacked_model, X_test, y_test)

stacked_model.save('Models/stacked_model')
print('>Saved stacked_model')

