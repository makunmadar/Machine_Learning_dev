## This script takes the model created from MultiOut_ML.py and applies it to an unseen test dataset
# Load in the test data
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
from joblib import dump, load

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
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_1000_S'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()},
                                           compile=False)
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

    chi_i = ((y_pred - y_true) / y_true) ** 2
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
drive_df['LF'] = drive_df['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
drive_df['error'] = drive_df['error'] * 2
upper_dri = np.log10(drive_df['LF'] + drive_df['error']) - np.log10(drive_df['LF'])
lower_dri = np.log10(drive_df['LF']) - np.log10(drive_df['LF'] - drive_df['error'])
drive_df['LF'] = np.log10(drive_df['LF'])

# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz_S'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file, usecols=range(13))

# print("Mean of y: ", np.mean(y_test, axis=0))
# print("Std of y: ", np.std(y_test, axis=0))

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
# scaler_feat = load('mm_scaler_feat.bin')
X_test = scaler_feat.transform(X_test)

# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
# scaler_label = load('std_scaler_label.bin')
y_test = scaler_label.transform(y_test)


# exit()

# Other half test sample
# X_test = X_test[1::2]
# y_test = y_test[1::2]

# Load a model from the Model directory
model_1 = tf.keras.models.load_model('Models/Ensemble_model_1_1000_S', compile=False)
#model_2 = tf.keras.models.load_model('Models/Ensemble_model_2_1000_S', compile=False)
# model_3 = tf.keras.models.load_model('Models/Ensemble_model_3_1000_S', compile=False)
# model_4 = tf.keras.models.load_model('Models/Ensemble_model_4_1000_S', compile=False)
# model_5 = tf.keras.models.load_model('Models/Ensemble_model_5_1000_S', compile=False)
#
# stacked_model = tf.keras.models.load_model('Models/stacked_model_1000_S', compile=False)

n_members = 1
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Make a prediction for test data
yhat_1 = model_1.predict(X_test)
# yhat_2 = model_2.predict(X_test)
# yhat_3 = model_3.predict(X_test)
# yhat_4 = model_4.predict(X_test)
# yhat_5 = model_5.predict(X_test)

ensamble_pred = list()
for model in members:
    yhat = model.predict(X_test)
    yhat = scaler_label.inverse_transform(yhat)
    ensamble_pred.append(yhat)

yhatavg = np.mean(ensamble_pred, axis=0)

# yhat_stacked = predict_stacked_model(stacked_model, X_test)

# De-normalize the predictions and truth data
yhat_1 = scaler_label.inverse_transform(yhat_1)

# yhat_2 = scaler_label.inverse_transform(yhat_2)

# yhat_3 = scaler_label.inverse_transform(yhat_3)
#
# yhat_4 = scaler_label.inverse_transform(yhat_4)
#
# yhat_5_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_5])
# yhat_5_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_5])
# yhat_5 = np.hstack([yhat_5_z, yhat_5_k])
#
# yhat_stacked_z = scaler_label_z.inverse_transform([i[0:13] for i in yhat_stacked])
# yhat_stacked_k = scaler_label_k.inverse_transform([i[13:22] for i in yhat_stacked])
# yhat_stacked = np.hstack([yhat_stacked_z, yhat_stacked_k])

y_test = scaler_label.inverse_transform(y_test)
# exit()

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

m = 138
for i in range(3):
    axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', label="1", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', label="2", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', label="3", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', label="4", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:13], yhatavg[i+m][0:13], 'k--', label="Avg ensemble")
    # # axs[i].plot(bins[0:13], yhat_stacked[i+m][0:13], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:13], y_test[i+m][0:13], 'gx-', label="True model "+str(i+1+m))
    axs[i].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                 ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    # axs[i+3].plot(bins[13:22], yhat_1[i+m][13:22], '--', label="1", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_2[i+m][13:22], '--', label="2", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_3[i+m][13:22], '--', label="3", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_4[i+m][13:22], '--', label="4", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_5[i+m][13:22], '--', label="5", alpha=0.5)
    # axs[i+3].plot(bins[13:22], yhatavg[i+m][13:22], 'k--', label="Avg ensemble")
    # # axs[i+3].plot(bins[13:22], yhat_stacked[i+m][13:22], 'b--', label="Stacked ensemble")
    # axs[i+3].plot(bins[13:22], y_test[i+m][13:22], 'gx-', label="True model " + str(i + 1 + m))
    # axs[i+3].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
    #                    ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
    # axs[i + 3].legend()
    # axs[i + 3].invert_xaxis()
    # axs[i + 3].set_xlabel("K-band magnitude", fontsize=16)
    # axs[i+3].set_xlim((-17.5, -26))

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel(r'Log$_{10}$(L$_{K}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 0
for i in range(3):
    axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', label="1", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', label="2", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', label="3", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', label="4", alpha=0.5)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', label="5", alpha=0.5)
    axs[i].plot(bins[0:13], yhatavg[i+m][0:13], 'k--', label="Avg ensemble")
    # # axs[i].plot(bins[0:13], yhat_stacked[i+m][0:13], 'b--', label="Stacked ensemble")
    axs[i].plot(bins[0:13], y_test[i+m][0:13], 'gx-', label="True model "+str(i+1+m))
    axs[i].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                 ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

    # axs[i+3].plot(bins[13:22], yhat_1[i+m][13:22], '--', label="1", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_2[i+m][13:22], '--', label="2", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_3[i+m][13:22], '--', label="3", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_4[i+m][13:22], '--', label="4", alpha=0.5)
    # # axs[i+3].plot(bins[13:22], yhat_5[i+m][13:22], '--', label="5", alpha=0.5)
    # axs[i+3].plot(bins[13:22], yhatavg[i+m][13:22], 'k--', label="Avg ensemble")
    # # axs[i+3].plot(bins[13:22], yhat_stacked[i+m][13:22], 'b--', label="Stacked ensemble")
    # axs[i+3].plot(bins[13:22], y_test[i+m][13:22], 'gx-', label="True model " + str(i + 1 + m))
    # axs[i+3].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
    #                    ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
    # axs[i + 3].legend()
    # axs[i + 3].invert_xaxis()
    # axs[i + 3].set_xlabel("K-band magnitude", fontsize=16)
    # axs[i+3].set_xlim((-17.5, -26))

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel(r'Log$_{10}$(L$_{K}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

# Model evaluation
predictions_1 = np.ravel(yhat_1)
predictions_2 = np.ravel(yhatavg)
# predictions_3 = np.ravel(yhat_stacked)
truth = np.ravel(y_test)

# Using MAE from sklearn
print('\n')
print('MAE combo single model: ', mean_absolute_error(truth, predictions_1))
print('MAE combo avg ensemble: ', mean_absolute_error(truth, predictions_2))
# print('MAE combo stacked: ', mean_absolute_error(truth, predictions_3))

predictions_1_z = np.ravel([i[0:13] for i in yhat_1])
predictions_2_z = np.ravel([i[0:13] for i in yhatavg])
# predictions_3_z = np.ravel([i[0:13] for i in yhat_stacked])
truth_z = np.ravel([i[0:13] for i in y_test])
print('\n')
print('MAE dn/dz single model: ', mean_absolute_error(truth_z, predictions_1_z))
print('MAE dn/dz avg ensemble: ', mean_absolute_error(truth_z, predictions_2_z))
# print('MAE dn/dz stacked: ', mean_absolute_error(truth_z, predictions_3_z))

predictions_1_k = np.ravel([i[13:22] for i in yhat_1])
predictions_2_k = np.ravel([i[13:22] for i in yhatavg])
# predictions_3_k = np.ravel([i[13:22] for i in yhat_stacked])
truth_k = np.ravel([i[13:22] for i in y_test])
print('\n')
print('MAE LF single model: ', mean_absolute_error(truth_k, predictions_1_k))
print('MAE LF avg ensemble: ', mean_absolute_error(truth_k, predictions_2_k))
# print('MAE LF stacked: ', mean_absolute_error(truth_k, predictions_3_k))
print('\n')

#
# # Plot the residuals
# fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
# fig.subplots_adjust(wspace=0)
# sns.residplot(x=predictions_2_z, y=truth_z, ax=axs, label="Avg ensemble")
# # sns.residplot(x=predictions_3_z, y=truth_z, ax=axs, label="Stacked ensemble")
# axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
# axs.set_xlabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
# plt.legend()
# plt.show()
#
# fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
# fig.subplots_adjust(wspace=0)
# sns.residplot(x=predictions_2_k, y=truth_k, ax=axs, label="Avg ensemble")
# # sns.residplot(x=predictions_3_k, y=truth_k, ax=axs, label="Stacked ensemble")
# axs.set_ylabel("Residuals (y-y$_p$)", fontsize=15)
# axs.set_xlabel(r"Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]", fontsize=15)
# plt.legend()
# plt.show()

# Manual MAE score
n = len(bins)
N = len(y_test)

MAE = []
# X_test = scaler_feat.inverse_transform(X_test)
ar = []
vhd = []
vhb = []
ah = []
ac = []
nsf = []

for j in range(N):
    maei = 0
    for i in range(n):
        sumi = abs(y_test[j][i] - yhatavg[j][i])
        maei += sumi
    maei = maei / n
    if maei > 0.3:
        print("Model ", j + 1, "had MAE: ", maei)
    MAE.append(maei)
    ar.append(X_test[j][0])
    vhd.append(X_test[j][1])
    vhb.append(X_test[j][2])
    ah.append(X_test[j][3])
    ac.append(X_test[j][4])
    nsf.append(X_test[j][5])

nbin = 12

binar = np.linspace(min(ar), max(ar), nbin)
dar = binar[1] - binar[0]
idxar = np.digitize(ar, binar)
MAE = np.array(MAE)
medar = [np.median(MAE[idxar == k]) for k in range(nbin)]
stdar = [MAE[idxar == k].std() for k in range(nbin)]

binvhd = np.linspace(min(vhd), max(vhd), nbin)
dvhd = binvhd[1] - binvhd[0]
idxvhd = np.digitize(vhd, binvhd)
medvhd = [np.median(MAE[idxvhd == k]) for k in range(nbin)]
stdvhd = [MAE[idxvhd == k].std() for k in range(nbin)]

binvhb = np.linspace(min(vhb), max(vhb), nbin)
dvhb = binvhb[1] - binvhb[0]
idxvhb = np.digitize(vhb, binvhb)
medvhb = [np.median(MAE[idxvhb == k]) for k in range(nbin)]
stdvhb = [MAE[idxvhb == k].std() for k in range(nbin)]

binah = np.linspace(min(ah), max(ah), nbin)
dah = binah[1] - binah[0]
idxah = np.digitize(ah, binah)
medah = [np.median(MAE[idxah == k]) for k in range(nbin)]
stdah = [MAE[idxah == k].std() for k in range(nbin)]

binac = np.linspace(min(ac), max(ac), nbin)
dac = binac[1] - binac[0]
idxac = np.digitize(ac, binac)
medac = [np.median(MAE[idxac == k]) for k in range(nbin)]
stdac = [MAE[idxac == k].std() for k in range(nbin)]

binnsf = np.linspace(min(nsf), max(nsf), nbin)
dnsf = binnsf[1] - binnsf[0]
idxnsf = np.digitize(nsf, binnsf)
mednsf = [np.median(MAE[idxnsf == k]) for k in range(nbin)]
stdnsf = [MAE[idxnsf == k].std() for k in range(nbin)]

plt.hist(MAE, bins=50)
plt.xlabel("Total MAE per test sample", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

axs[0].plot(ar, MAE, '.')
axs[0].errorbar(binar - dar / 2, medar, stdar, marker='s', color='black', alpha=0.7, label="Median")
axs[0].set_ylabel("MAE per test sample", fontsize=16)
axs[0].set_xlabel("Alpha reheat", fontsize=16)
axs[0].legend()
axs[1].plot(vhd, MAE, '.')
axs[1].errorbar(binvhd - dvhd / 2, medvhd, stdvhd, marker='s', color='black', alpha=0.7, label="Median")
axs[1].set_xlabel("Vhotdisk", fontsize=16)
axs[1].legend()
axs[2].plot(vhb, MAE, '.')
axs[2].errorbar(binvhb - dvhb / 2, medvhb, stdvhb, marker='s', color='black', alpha=0.7, label="Median")
axs[2].set_xlabel("Vhotbust", fontsize=16)
axs[2].legend()

axs[3].plot(ah, MAE, '.')
axs[3].errorbar(binah - dah / 2, medah, stdah, marker='s', color='black', alpha=0.7, label="Median")
axs[3].set_ylabel("MAE per test sample", fontsize=16)
axs[3].set_xlabel("Alpha hot", fontsize=16)
axs[3].legend()
axs[4].plot(ac, MAE, '.')
axs[4].errorbar(binac - dac / 2, medac, stdac, marker='s', color='black', alpha=0.7, label="Median")
axs[4].set_xlabel("Alpha cool", fontsize=16)
axs[4].legend()
axs[5].plot(nsf, MAE, '.')
axs[5].errorbar(binnsf - dnsf / 2, mednsf, stdnsf, marker='s', color='black', alpha=0.7, label="Median")
axs[5].set_xlabel("Nu SF", fontsize=16)
axs[5].legend()

plt.show()
