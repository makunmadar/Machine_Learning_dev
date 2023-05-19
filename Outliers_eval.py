import numpy as np
from numpy import genfromtxt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Make a prediction with the stacked model
def predict_stacked_model(model, inputX):
    # Prepare input data
    X = [inputX for _ in range(len(model.input))]
    # Make prediction
    return model.predict(X)


# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz'

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

model_2 = tf.keras.models.load_model('Models/stacked_model_200', compile=False)
model_4 = tf.keras.models.load_model('Models/stacked_model_400', compile=False)
model_6 = tf.keras.models.load_model('Models/stacked_model_600', compile=False)
model_8 = tf.keras.models.load_model('Models/stacked_model_800', compile=False)
model_10 = tf.keras.models.load_model('Models/stacked_model_1000', compile=False)

yhat_2 = predict_stacked_model(model_2, X_test)
yhat_4 = predict_stacked_model(model_4, X_test)
yhat_6 = predict_stacked_model(model_6, X_test)
yhat_8 = predict_stacked_model(model_8, X_test)
yhat_10 = predict_stacked_model(model_10, X_test)

yhat_2 = scaler_label.inverse_transform(yhat_2)
yhat_4 = scaler_label.inverse_transform(yhat_4)
yhat_6 = scaler_label.inverse_transform(yhat_6)
yhat_8 = scaler_label.inverse_transform(yhat_8)
yhat_10 = scaler_label.inverse_transform(yhat_10)

y_test = scaler_label.inverse_transform(y_test)

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#fig.subplots_adjust(wspace=0)

axs[0,0].plot(bins[0:12], yhat_2[3][0:12], '--', label="200 model")
axs[0,0].plot(bins[0:12], yhat_4[3][0:12], '--', label="400 model")
axs[0,0].plot(bins[0:12], yhat_6[3][0:12], '--', label="600 model")
axs[0,0].plot(bins[0:12], yhat_8[3][0:12], '--', label="800 model")
axs[0,0].plot(bins[0:12], yhat_10[3][0:12], '--', label="1000 model")
axs[0,0].plot(bins[0:12], y_test[3][0:12], 'gx-', label="True model")
axs[0,0].legend()
axs[0,0].set_xlabel("Redshift, z", fontsize=16)
axs[0,0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[0,1].plot(bins[0:12], yhat_2[7][0:12], '--', label="200 model")
axs[0,1].plot(bins[0:12], yhat_4[7][0:12], '--', label="400 model")
axs[0,1].plot(bins[0:12], yhat_6[7][0:12], '--', label="600 model")
axs[0,1].plot(bins[0:12], yhat_8[7][0:12], '--', label="800 model")
axs[0,1].plot(bins[0:12], yhat_10[7][0:12], '--', label="1000 model")
axs[0,1].plot(bins[0:12], y_test[7][0:12], 'gx-', label="True model")
axs[0,1].legend()
axs[0,1].set_xlabel("Redshift, z", fontsize=16)
axs[0,1].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[1,0].plot(bins[12:24], yhat_2[3][12:24], '--', label="200 model")
axs[1,0].plot(bins[12:24], yhat_4[3][12:24], '--', label="400 model")
axs[1,0].plot(bins[12:24], yhat_6[3][12:24], '--', label="600 model")
axs[1,0].plot(bins[12:24], yhat_8[3][12:24], '--', label="800 model")
axs[1,0].plot(bins[12:24], yhat_10[3][12:24], '--', label="1000 model")
axs[1,0].plot(bins[12:24], y_test[3][12:24], 'gx-', label="True model")
axs[1,0].legend()
axs[1,0].invert_xaxis()
axs[1,0].set_xlabel("K-band magnitude", fontsize=16)
axs[1,0].set_ylim((-6, -1))
axs[1,0].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)

axs[1,1].plot(bins[12:24], yhat_2[7][12:24], '--', label="200 model")
axs[1,1].plot(bins[12:24], yhat_4[7][12:24], '--', label="400 model")
axs[1,1].plot(bins[12:24], yhat_6[7][12:24], '--', label="600 model")
axs[1,1].plot(bins[12:24], yhat_8[7][12:24], '--', label="800 model")
axs[1,1].plot(bins[12:24], yhat_10[7][12:24], '--', label="1000 model")
axs[1,1].plot(bins[12:24], y_test[7][12:24], 'gx-', label="True model")
axs[1,1].legend()
axs[1,1].invert_xaxis()
axs[1,1].set_xlabel("K-band magnitude", fontsize=16)
axs[1,1].set_ylim((-6, -1))
axs[1,1].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()