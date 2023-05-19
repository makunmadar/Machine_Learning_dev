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


def load_all_models(n_models):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_'+str(i+1)+'_200'
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

# Other half test sample
# X_test = X_test[1::2]
# y_test = y_test[1::2]

# Load a model from the Model directory
model_1 = tf.keras.models.load_model('Models/Ensemble_model_1_200', compile=False)
model_2 = tf.keras.models.load_model('Models/Ensemble_model_2_200', compile=False)
model_3 = tf.keras.models.load_model('Models/Ensemble_model_3_200', compile=False)
model_4 = tf.keras.models.load_model('Models/Ensemble_model_4_200', compile=False)
model_5 = tf.keras.models.load_model('Models/Ensemble_model_5_200', compile=False)

stacked_model = tf.keras.models.load_model('Models/stacked_model_200', compile=False)

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
yhat_1 = model_1.predict(X_test)
yhat_2 = model_2.predict(X_test)
yhat_3 = model_3.predict(X_test)
yhat_4 = model_4.predict(X_test)
yhat_5 = model_5.predict(X_test)

ensamble_pred = list()
for model in members:
    yhat = model.predict(X_test)
    yhat = scaler_label.inverse_transform(yhat)
    ensamble_pred.append(yhat)

yhatavg = np.mean(ensamble_pred, axis=0)

yhat_stacked = predict_stacked_model(stacked_model, X_test)

# De-normalize the predictions and truth data
yhat_1 = scaler_label.inverse_transform(yhat_1)
yhat_2 = scaler_label.inverse_transform(yhat_2)
yhat_3 = scaler_label.inverse_transform(yhat_3)
yhat_4 = scaler_label.inverse_transform(yhat_4)
yhat_5 = scaler_label.inverse_transform(yhat_5)

yhat_stacked = scaler_label.inverse_transform(yhat_stacked)

y_test = scaler_label.inverse_transform(y_test)
# print('Predicted: %s' % yhat[1])
# print('True: %s' % y_test[1])

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

for i in range(3):

    axs[i].plot(bins[0:12], yhat_1[i][0:12], '--', label="1", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_2[i][0:12], '--', label="2", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_3[i][0:12], '--', label="3", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_4[i][0:12], '--', label="4", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_5[i][0:12], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:12], yhatavg[i][0:12], 'k--', label="Avg ensemble")
    axs[i].plot(bins[0:12], yhat_stacked[i][0:12], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:12], y_test[i][0:12], 'gx-', label="True model "+str(i+1))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    axs[i+3].plot(bins[12:24], yhat_1[i][12:24], '--', label="1", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_2[i][12:24], '--', label="2", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_3[i][12:24], '--', label="3", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_4[i][12:24], '--', label="4", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_5[i][12:24], '--', label="5", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhatavg[i][12:24], 'k--', label="Avg ensemble")
    axs[i+3].plot(bins[12:24], yhat_stacked[i][12:24], 'b--', label="Stacked ensemble")
    axs[i+3].plot(bins[12:24], y_test[i][12:24], 'gx-', label="True model "+str(i+1))
    axs[i+3].legend()
    axs[i+3].invert_xaxis()
    axs[i+3].set_xlabel("K-band magnitude", fontsize=16)
    axs[i+3].set_ylim((-6, 0))

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

for i in range(3):

    axs[i].plot(bins[0:12], yhat_1[i+6][0:12], '--', label="1", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_2[i+6][0:12], '--', label="2", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_3[i+6][0:12], '--', label="3", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_4[i+6][0:12], '--', label="4", alpha=0.5)
    axs[i].plot(bins[0:12], yhat_5[i+6][0:12], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:12], yhatavg[i+6][0:12], 'k--', label="Avg ensemble")
    axs[i].plot(bins[0:12], yhat_stacked[i+6][0:12], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:12], y_test[i+6][0:12], 'gx-', label="True model "+str(i+1))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    axs[i+3].plot(bins[12:24], yhat_1[i+6][12:24], '--', label="1", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_2[i+6][12:24], '--', label="2", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_3[i+6][12:24], '--', label="3", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_4[i+6][12:24], '--', label="4", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhat_5[i+6][12:24], '--', label="5", alpha=0.5)
    axs[i+3].plot(bins[12:24], yhatavg[i+6][12:24], 'k--', label="Avg ensemble")
    axs[i+3].plot(bins[12:24], yhat_stacked[i+6][12:24], 'b--', label="Stacked ensemble")
    axs[i+3].plot(bins[12:24], y_test[i+6][12:24], 'gx-', label="True model "+str(i+1))
    axs[i+3].legend()
    axs[i+3].invert_xaxis()
    axs[i+3].set_xlabel("K-band magnitude", fontsize=16)
    axs[i+3].set_ylim((-6, 0))

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()


# Model evaluation
predictions_1 = np.ravel(yhat_1)
predictions_2 = np.ravel(yhatavg)
predictions_3 = np.ravel(yhat_stacked)
truth = np.ravel(y_test)

# Using MAE from sklearn
print('\n')
print('MAE combo single model: ', mean_absolute_error(truth, predictions_1))
print('MAE combo avg ensemble: ', mean_absolute_error(truth, predictions_2))
print('MAE combo stacked: ', mean_absolute_error(truth, predictions_3))

predictions_1_z = np.ravel([i[0:12] for i in yhat_1])
predictions_2_z = np.ravel([i[0:12] for i in yhatavg])
predictions_3_z = np.ravel([i[0:12] for i in yhat_stacked])
truth_z = np.ravel([i[0:12] for i in y_test])
print('\n')
print('MAE dn/dz single model: ', mean_absolute_error(truth_z, predictions_1_z))
print('MAE dn/dz avg ensemble: ', mean_absolute_error(truth_z, predictions_2_z))
print('MAE dn/dz stacked: ', mean_absolute_error(truth_z, predictions_3_z))

predictions_1_k = np.ravel([i[12:24] for i in yhat_1])
predictions_2_k = np.ravel([i[12:24] for i in yhatavg])
predictions_3_k = np.ravel([i[12:24] for i in yhat_stacked])
truth_k = np.ravel([i[12:24] for i in y_test])
print('\n')
print('MAE LF single model: ', mean_absolute_error(truth_k, predictions_1_k))
print('MAE LF avg ensemble: ', mean_absolute_error(truth_k, predictions_2_k))
print('MAE LF stacked: ', mean_absolute_error(truth_k, predictions_3_k))

# Plot the residuals
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0)
sns.residplot(x=predictions_2_z, y=truth_z, ax=axs, label="Avg ensemble")
sns.residplot(x=predictions_3_z, y=truth_z, ax=axs, label="Stacked ensemble")
axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
axs.set_xlabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
plt.legend()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0)
sns.residplot(x=predictions_2_k, y=truth_k, ax=axs, label="Avg ensemble")
sns.residplot(x=predictions_3_k, y=truth_k, ax=axs, label="Stacked ensemble")
axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
axs.set_xlabel(r"Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]", fontsize=15)
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
