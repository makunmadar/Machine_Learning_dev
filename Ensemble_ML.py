import tensorflow as tf
from keras.layers import concatenate
from keras.layers import Dense
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# Define stacked model from multiple member input models
def define_stacked_model(members):
    # Update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # Make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # Define multi-headed input
    ensemble_visible = [model.input for model in members]
    # Concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(64, activation='sigmoid')(merge)
    output = Dense(24)(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='Plots/stacked.png')
    # Compile
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
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

# Fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # Fit model
    log_dir = "logs/fit/stacked_model"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X, inputy,
              validation_split=0.2,
              epochs=300, verbose=0, callbacks=[early_stopping, tensorboard_callback])

    model.save('Models/stacked_model')

# Load the models
keras_model = tf.keras.models.load_model('Models/Ensemble_model_1', compile=False)
keras_model._name = 'model1'
keras_model2 = tf.keras.models.load_model('Models/Ensemble_model_2', compile=False)
keras_model2._name = 'model2'
keras_model3 = tf.keras.models.load_model('Models/Ensemble_model_3', compile=False)
keras_model3._name = 'model3'
keras_model4 = tf.keras.models.load_model('Models/Ensemble_model_4', compile=False)
keras_model4._name = 'model4'
keras_model5 = tf.keras.models.load_model('Models/Ensemble_model_5', compile=False)
keras_model5._name = 'model5'

models = [keras_model, keras_model2, keras_model3, keras_model4, keras_model5]

stacked_model = define_stacked_model(models)

# Import the test data
feature_file = 'Data/Data_for_ML/training_data/feature'
label_file = 'Data/Data_for_ML/training_data/label_sub12_dndz'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# One half test sample
# X_test = X_test[0::2]
# y_test = y_test[0::2]

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test = scaler_label.transform(y_test)

fit_stacked_model(stacked_model, X_test, y_test)