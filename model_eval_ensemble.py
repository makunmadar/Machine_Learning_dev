## This script takes the model created from MultiOut_ML.py and applies it to an unseen test dataset
# Load in the test data
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd

print(tf.version.VERSION)


def kband_df(path, columns):
    '''
    This function extracts just the k_band LF data and saves it in a dataframe.

    :param path: path to the LF file
    :param columns: what are the names of the magnitude columns?
    :return: dataframe
    '''
    data = []
    with open(path, 'r') as fh:
        for curline in fh:
            if curline.startswith("#"):
                header = curline
            else:
                row = curline.strip().split()
                data.append(row)

    data = np.vstack(data)
    df = pd.DataFrame(data=data)
    df = df.apply(pd.to_numeric)
    df.columns = columns

    return df

def load_all_models(n_models):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_'+str(i+1)+'_1000_S'
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


# Import the observational data
# Import the Bagley et al. 2020 data from .csv file
bag_df = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Corrected_Ha_Bagley_redshift.csv", delimiter=',')
upper_bag = np.log10(bag_df["+"]) - np.log10(bag_df["y"])
lower_bag = np.log10(bag_df["y"]) - np.log10(bag_df["-"])
bag_df["y"] = np.log10(bag_df["y"])

# Import the Driver et al. 2012 data from .data file
columns_d = ['Mag', 'LF', 'error', 'Freq']
drive_df = kband_df("Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data", columns_d)
drive_df = drive_df[(drive_df != 0).all(1)]
drive_df['LF'] = drive_df['LF']*2 # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
drive_df['error'] = drive_df['error']*2
upper_dri = np.log10(drive_df['LF'] + drive_df['error']) - np.log10(drive_df['LF'])
lower_dri = np.log10(drive_df['LF']) - np.log10(drive_df['LF'] - drive_df['error'])
drive_df['LF'] = np.log10(drive_df['LF'])

# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz_S'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

y_testz = [i[0:13] for i in y_test]
y_testk = [i[13:22] for i in y_test]

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)

# Use standard scalar for the label data
scaler_label_z = StandardScaler()
scaler_label_k = StandardScaler()
scaler_label_z.fit(y_testz)
y_testz = scaler_label_z.transform(y_testz)

scaler_label_k.fit(y_testk)
y_testk = scaler_label_k.transform(y_testk)

y_test = np.hstack([y_testz, y_testk])

# Other half test sample
# X_test = X_test[1::2]
# y_test = y_test[1::2]

# Load a model from the Model directory
model_1 = tf.keras.models.load_model('Models/Ensemble_model_1_1000_S', compile=False)
model_2 = tf.keras.models.load_model('Models/Ensemble_model_2_1000_S', compile=False)
model_3 = tf.keras.models.load_model('Models/Ensemble_model_3_1000_S', compile=False)
model_4 = tf.keras.models.load_model('Models/Ensemble_model_4_1000_S', compile=False)
model_5 = tf.keras.models.load_model('Models/Ensemble_model_5_1000_S', compile=False)

stacked_model = tf.keras.models.load_model('Models/stacked_model_1000_S', compile=False)

n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Make a prediction for test data
yhat_1 = model_1.predict(X_test)
yhat_2 = model_2.predict(X_test)
yhat_3 = model_3.predict(X_test)
yhat_4 = model_4.predict(X_test)
yhat_5 = model_5.predict(X_test)

ensamble_pred = list()
for model in members:
    yhat = model.predict(X_test)
    yhat_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat])
    yhat_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat])
    yhat = np.hstack([yhat_z, yhat_k])
    ensamble_pred.append(yhat)

yhatavg = np.mean(ensamble_pred, axis=0)

yhat_stacked = predict_stacked_model(stacked_model, X_test)

# De-normalize the predictions and truth data
yhat_1_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_1])
yhat_1_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_1])
yhat_1 = np.hstack([yhat_1_z, yhat_1_k])

yhat_2_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_2])
yhat_2_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_2])
yhat_2 = np.hstack([yhat_2_z, yhat_2_k])

yhat_3_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_3])
yhat_3_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_3])
yhat_3 = np.hstack([yhat_3_z, yhat_3_k])

yhat_4_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_4])
yhat_4_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_4])
yhat_4 = np.hstack([yhat_4_z, yhat_4_k])

yhat_5_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_5])
yhat_5_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_5])
yhat_5 = np.hstack([yhat_5_z, yhat_5_k])

yhat_stacked_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_stacked])
yhat_stacked_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_stacked])
yhat_stacked = np.hstack([yhat_stacked_z, yhat_stacked_k])

y_test_z = scaler_label_z.inverse_transform([i[0:13] for i in y_test])
y_test_k = scaler_label_k.inverse_transform([i[13:22] for i in y_test])
y_test = np.hstack([y_test_z, y_test_k])
# print('Predicted: %s' % yhat[1])
# print('True: %s' % y_test[1])

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

# Ignore points where y_true = -20
# y_truek = list()
# yhat_1k = list()
# yhat_2k = list()
# yhat_3k = list()
# yhat_4k = list()
# yhat_5k = list()
# yhatavg_k = list()
# yhatstack_k = list()
# binsk = list()
#
# for i in range(200):
#     y_tk = y_test_k[i]
#     ytk = y_tk[y_tk > -100]
#     y_truek.append(ytk)
#
#     y1k = yhat_1_k[i][y_tk > -100]
#     yhat_1k.append(y1k)
#     y2k = yhat_2_k[i][y_tk > -100]
#     yhat_2k.append(y2k)
#     y3k = yhat_3_k[i][y_tk > -100]
#     yhat_3k.append(y3k)
#     y4k = yhat_4_k[i][y_tk > -100]
#     yhat_4k.append(y4k)
#     y5k = yhat_5_k[i][y_tk > -100]
#     yhat_5k.append(y5k)
#     yak = yhat_avg_k[i][y_tk > -100]
#     yhatavg_k.append(yak)
#     ysk = yhat_stacked_k[i][y_tk > -100]
#     yhatstack_k.append(ysk)
#
#     bins_k = bins[12:24]
#     bk = bins_k[y_tk > -100]
#     binsk.append(bk)

# print('Min ytrue_k: ', [min(a) for a in y_truek])
# print('Min yhatstack_k: ', [min(a) for a in yhatstack_k])

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m=151
for i in range(3):

    axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', label="1", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', label="2", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', label="3", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', label="4", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:13], yhatavg[i+m][0:13], 'k--', label="Avg ensemble")
    axs[i].plot(bins[0:13], yhat_stacked[i+m][0:13], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:13], y_test[i+m][0:13], 'gx-', label="True model "+str(i+1+m))
    axs[i].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
    axs[i].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    axs[i+3].plot(bins[13:22], yhat_1[i+m][13:22], '--', label="1", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_2[i+m][13:22], '--', label="2", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_3[i+m][13:22], '--', label="3", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_4[i+m][13:22], '--', label="4", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_5[i+m][13:22], '--', label="5", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhatavg[i+m][13:22], 'k--', label="Avg ensemble")
    axs[i+3].plot(bins[13:22], yhat_stacked[i+m][13:22], 'b--', label="Stacked ensemble")
    axs[i+3].plot(bins[13:22], y_test[i][13:22], 'gx-', label="True model "+str(i+1+m))
    #axs[i+3].plot(binsk[i+m], y_truek[i+m], 'gx-', label="Trimmed true "+str(i+1+m))
    axs[i+3].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
    axs[i+3].legend()
    axs[i+3].invert_xaxis()
    axs[i+3].set_xlabel("K-band magnitude", fontsize=16)
    #axs[i+3].set_xlim((-17.5, -26))

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

for i in range(3):

    axs[i].plot(bins[0:13], yhat_1[i+6][0:13], '--', label="1", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_2[i+6][0:13], '--', label="2", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_3[i+6][0:13], '--', label="3", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_4[i+6][0:13], '--', label="4", alpha=0.5)
    axs[i].plot(bins[0:13], yhat_5[i+6][0:13], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:13], yhatavg[i+6][0:13], 'k--', label="Avg ensemble")
    axs[i].plot(bins[0:13], yhat_stacked[i+6][0:13], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:13], y_test[i+6][0:13], 'gx-', label="True model "+str(i+1))
    axs[i].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                    ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
    axs[i].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    axs[i+3].plot(bins[13:22], yhat_1[i+6][13:22], '--', label="1", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_2[i+6][13:22], '--', label="2", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_3[i+6][13:22], '--', label="3", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_4[i+6][13:22], '--', label="4", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhat_5[i+6][13:22], '--', label="5", alpha=0.5)
    axs[i+3].plot(bins[13:22], yhatavg[i+6][13:22], 'k--', label="Avg ensemble")
    axs[i+3].plot(bins[13:22], yhat_stacked[i+6][13:22], 'b--', label="Stacked ensemble")
    axs[i+3].plot(bins[13:22], y_test[i+6][13:22], 'gx-', label="True model "+str(i+1+6))
    #axs[i+3].plot(binsk[i+6], y_truek[i+6], 'gx-', label="True model "+str(i+1+6))
    axs[i+3].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
    axs[i+3].legend()
    axs[i+3].invert_xaxis()
    axs[i+3].set_xlabel("K-band magnitude", fontsize=16)
    #axs[i+3].set_xlim((-17.5, -26))

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

predictions_1_z = np.ravel([i[0:13] for i in yhat_1])
predictions_2_z = np.ravel([i[0:13] for i in yhatavg])
predictions_3_z = np.ravel([i[0:13] for i in yhat_stacked])
truth_z = np.ravel([i[0:13] for i in y_test])
print('\n')
print('MAE dn/dz single model: ', mean_absolute_error(truth_z, predictions_1_z))
print('MAE dn/dz avg ensemble: ', mean_absolute_error(truth_z, predictions_2_z))
print('MAE dn/dz stacked: ', mean_absolute_error(truth_z, predictions_3_z))

# predictions_1_k = np.hstack(yhat_1k)
# predictions_2_k = np.hstack(yhatavg_k)
# predictions_3_k = np.hstack(yhatstack_k)
# truth_k = np.hstack(y_truek)
predictions_1_k = np.ravel([i[13:22] for i in yhat_1])
predictions_2_k = np.ravel([i[13:22] for i in yhatavg])
predictions_3_k = np.ravel([i[13:22] for i in yhat_stacked])
truth_k = np.ravel([i[13:22] for i in y_test])
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
