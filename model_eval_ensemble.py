"""
Evaluation of multiple tensorflow models ensembled, applying it to an unseen test dataset seperately saved.
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

def load_all_models(n_models):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_'+str(i+1)
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()}, compile=False)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models

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
label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# Other half test sample
# X_test = X_test[0::2]
# y_test = y_test[0::2]

# Load a model from the Model directory
#model_1 = tf.keras.models.load_model('Models/model_124_124_124_ES_T400_LR0.005', custom_objects={'rmse': rmse}, compile=False)
stacked_model = tf.keras.models.load_model('Models/stacked_model', compile=False)

n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test = scaler_label.transform(y_test)

# Make a prediction for test data
#yhat_1 = model_1.predict(X_test)

ensamble_pred = list()
for model in members:
    yhat = model.predict(X_test)
    yhat = scaler_label.inverse_transform(yhat)
    ensamble_pred.append(yhat)

yhatavg = np.mean(ensamble_pred, axis=0)

yhat_stacked = stacked_model.predict(X_test)

# De-normalize the predictions and truth data
#yhat_1 = scaler_label.inverse_transform(yhat_1)
yhat_stacked = scaler_label.inverse_transform(yhat_stacked)

y_test = scaler_label.inverse_transform(y_test)
# print('Predicted: %s' % yhat[1])
# print('True: %s' % y_test[1])

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

# Plot the results
fig, axs = plt.subplots(3, 3, figsize=(13, 15),
                        facecolor='w', edgecolor='k', sharey=True, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0)
axs = axs.ravel()

for i in range(9):

    #axs[i].plot(bins, yhat_1[i], 'x--', label="124 400 train ES LR0.005")
    axs[i].plot(bins, yhatavg[i], 'x--', label="Avg ensemble", alpha=0.5)
    axs[i].plot(bins, yhat_stacked[i], 'x--', label="Integrated stacked model", alpha=0.5)
    #axs[i].plot(bins, yhat_4[i], 'x--', label="LR: exp decay", alpha=0.5)

    axs[i].plot(bins, y_test[i], 'gx-', label="True model "+str(i+1))
    axs[i].legend()

fig.supylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
fig.supxlabel("Redshift, z", fontsize=16)
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(13, 15),
                        facecolor='w', edgecolor='k', sharey=True, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0)
axs = axs.ravel()

for i in range(9):

    #axs[i].plot(bins, yhat_1[i], 'x--', label="124 400 train ES LR0.005")
    axs[i].plot(bins, yhatavg[i+95], 'x--', label="Avg ensemble", alpha=0.5)
    axs[i].plot(bins, yhat_stacked[i], 'x--', label="Integrated stacked model", alpha=0.5)
    #axs[i].plot(bins, yhat_4[i], 'x--', label="LR: exp decay", alpha=0.5)

    axs[i].plot(bins, y_test[i+95], 'gx-', label="True model "+str(i+95+1))
    axs[i].legend()

fig.supylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
fig.supxlabel("Redshift, z", fontsize=16)
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(13, 15),
                        facecolor='w', edgecolor='k', sharey=True, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0)
axs = axs.ravel()

for i in range(9):

    #axs[i].plot(bins, yhat_1[i], 'x--', label="124 400 train ES LR0.005")
    axs[i].plot(bins, yhatavg[i+191], 'x--', label="Avg ensemble", alpha=0.5)
    #axs[i].plot(bins, yhat_stacked[i], 'x--', label="Integrated stacked model", alpha=0.5)
    #axs[i].plot(bins, yhat_4[i], 'x--', label="LR: exp decay", alpha=0.5)

    axs[i].plot(bins, y_test[i+191], 'gx-', label="True model "+str(i+191+1))
    axs[i].legend()

fig.supylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
fig.supxlabel("Redshift, z", fontsize=16)
plt.show()

# Model evaluation
# Using RMSE from tensorflow
#predictions_1 = np.ravel(yhat_1)
predictions_2 = np.ravel(yhatavg)
predictions_3 = np.ravel(yhat_stacked)
truth = np.ravel(y_test)
print('\n')
#print('RMSE of predictions_1: ', rmse(truth, predictions_1).numpy())
print('RMSE of predictions avg ensemble: ', rmse(truth, predictions_2).numpy())
print('RMSE of predictions stacked: ', rmse(truth, predictions_3).numpy())
# Using MAE from sklearn
print('\n')
#print('MAE of predictions_1: ', mean_absolute_error(truth, predictions_1))
print('MAE of predictions avg ensemble: ', mean_absolute_error(truth, predictions_2))
print('MAE of predictions stacked: ', mean_absolute_error(truth, predictions_3))

# Plot the residuals
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0)
#sns.residplot(x=yhat_counts, y=y_test_counts, ax=axs[0])
#sns.residplot(x=predictions_1, y=truth, ax=axs, label="512 400 train ES LR0.005")
sns.residplot(x=predictions_2, y=truth, ax=axs, label="Avg ensemble")
sns.residplot(x=predictions_3, y=truth, ax=axs, label="Integrated stacked model")
axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
#axs.set_xlabel("Log$_{10}$(Flux) [10$^{-16}$erg s$^{-1}$ cm$^{-2}$]", fontsize=15)
axs.set_xlabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
plt.legend()
plt.show()

# Perform a chi square test on the predictions
# print('\n')
# for i in range(len(y_test)):
#     chi = chi_test(y_test[i], yhat[i])
#     #chi_half = chi_test(y_test[i], yhat_half[i])
#
#     print(f'Test model {i+1} chi square score: ', chi)
#     print(f'Test half model {i+1} chi square score: ', chi_half)