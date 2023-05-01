"""
Evaluation of a tensorflow model, applying it to an unseen test dataset seperately saved.
Using a variety of metrics to assess the precision of predictions.
"""

## This script takes the model created from MultiOut_ML.py and applies it to an unseen test dataset
# Load in the test data
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(tf.version.VERSION)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def chi_test(y_true, y_pred):
    '''
    Perform a chi square test on the predicted data against the true values

    :param y_true: array of true model values
    :param y_pred: array of predicted model values
    :return: calculated absolute chi square value
    '''

    chi_i = ((y_pred - y_true) / y_true)**2
    chi_sum = sum(chi_i)

    return chi_sum


# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub6_dndz'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# Load a model from the Model directory
model_1 = tf.keras.models.load_model('Models/model_512_512_epoch150_T400', custom_objects={'rmse': rmse}, compile=False)
model_1.summary()

# Load the other models
model_2 = tf.keras.models.load_model('Models/model_512_512_epoch200_T400', custom_objects={'rmse': rmse}, compile=False)
#model_3 = tf.keras.models.load_model('Models/model_64_64_sigmoid', custom_objects={'rmse': rmse}, compile=False)
#model_4 = tf.keras.models.load_model('Models/model_124_124_sigmoid_LRexp', custom_objects={'rmse': rmse}, compile=False)


# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test = scaler_label.transform(y_test)

# Make a prediction for test data
yhat_1 = model_1.predict(X_test)
yhat_2 = model_2.predict(X_test)
#yhat_3 = model_3.predict(X_test)
#yhat_4 = model_4.predict(X_test)

# De-normalize the predictions and truth data
yhat_1 = scaler_label.inverse_transform(yhat_1)
yhat_2 = scaler_label.inverse_transform(yhat_2)
#yhat_64 = scaler_label.inverse_transform(yhat_64)
#yhat_exp = scaler_label.inverse_transform(yhat_exp)

y_test = scaler_label.inverse_transform(y_test)
# print('Predicted: %s' % yhat[1])
# print('True: %s' % y_test[1])

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub6_dndz'
bins = genfromtxt(bin_file)

# Plot the results
fig, axs = plt.subplots(3, 3, figsize=(13, 15),
                        facecolor='w', edgecolor='k', sharey=True, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0)
axs = axs.ravel()

for i in range(9):

    axs[i].plot(bins, yhat_1[i], 'x--', label="512 400 training epoch 150")
    axs[i].plot(bins, yhat_2[i], 'x--', label="512 400 training epoch 200", alpha=0.5)
    #axs[i].plot(bins, yhat_3[i], 'x--', label="64 64", alpha=0.5)
    #axs[i].plot(bins, yhat_4[i], 'x--', label="LR: exp decay", alpha=0.5)

    axs[i].plot(bins, y_test[i], 'gx-', label="True")
    axs[i].legend()

fig.supylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
fig.supxlabel("Redshift, z", fontsize=16)
plt.show()

# Model evaluation
# Using RMSE from tensorflow
predictions = np.ravel(yhat)
#half_predictions = np.ravel(yhat_half)
truth = np.ravel(y_test)
print('RMSE of predictions: ', rmse(truth, predictions).numpy())
#print('RMSE of half predictions: ', rmse(truth, half_predictions).numpy())

# Using MAE from sklearn
print('MAE of predictions: ', mean_absolute_error(truth, predictions))
#print('MAE of half predictions: ', mean_absolute_error(truth, half_predictions))

# Plot the residuals
#yhat_counts = np.ravel(yhat[:, 0:6])
yhat_dndz_rav = np.ravel(yhat)
#yhat_half_dndz_rav = np.ravel(yhat_half)
#y_test_counts = np.ravel(y_test[:, 0:6])
y_test_dndz_rav = np.ravel(y_test)

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0)
#sns.residplot(x=yhat_counts, y=y_test_counts, ax=axs[0])
sns.residplot(x=yhat_dndz_rav, y=y_test_dndz_rav, ax=axs, label="Full model")
#sns.residplot(x=yhat_half_dndz_rav, y=y_test_dndz_rav, ax=axs, label="Half model")
axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
#axs.set_xlabel("Log$_{10}$(Flux) [10$^{-16}$erg s$^{-1}$ cm$^{-2}$]", fontsize=15)
axs.set_xlabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
plt.show()

# Perform a chi square test on the predictions
print('\n')
for i in range(len(y_test)):
    chi = chi_test(y_test[i], yhat[i])
    #chi_half = chi_test(y_test[i], yhat_half[i])

    print(f'Test model {i+1} chi square score: ', chi)
    #print(f'Test half model {i+1} chi square score: ', chi_half)

