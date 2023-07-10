import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error


def load_all_models(n_models, X_test):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
           X_test: test sample in np.array already normalized
    :return: list of ensemble models
    """

    all_yhat = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_2512_mask'
        # Load model from file
        model = tf.keras.models.load_model(filename, compile=False)
        # Produce prediction
        yhat = model.predict(X_test)
        all_yhat.append(yhat)
        print('>loaded %s' % filename)

    return all_yhat

X_test = np.load('Data/Data_for_ML/testing_data/X_test.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test.npy')

scaler_feat = load('mm_scaler_feat.bin')
X_test = scaler_feat.transform(X_test)
# scaler_label = load('std_scaler_label.bin')

# Load all the models and make predictions on the test set
yhat_all = load_all_models(n_models=5, X_test=X_test)
yhat_avg = np.mean(yhat_all, axis=0)

# Individual predictions for plotting
yhat_1 = yhat_all[0]
yhat_2 = yhat_all[1]
yhat_3 = yhat_all[2]
yhat_4 = yhat_all[3]
yhat_5 = yhat_all[4]

bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)
bins_k = bins[13:22]

# Manual redshift distribution MAE score
y_testz = [i[0:13] for i in y_test]
yhatz = [i[0:13] for i in yhat_avg]

MAEz = []
for j in range(200):
    maei = mean_absolute_error(y_testz[j], yhatz[j])
    if maei > 0.2:
        print("Model ", j + 1, "had dn/dz MAE: ", maei)
    MAEz.append(maei)

# Manual luminosity function MAE score
y_testk = [i[13:22] for i in y_test]
yhatk = [i[13:22] for i in yhat_avg]

yhatk_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk, y_testk)]
binsk = []
for i in range(200):
    bk = bins_k[y_testk[i] != 0]
    binsk.append(bk)
y_testk = [row[row != 0] for row in y_testk]

MAEk = []
for j in range(200):
    maei = mean_absolute_error(y_testk[j], yhatk_mae[j])
    if maei > 0.2:
        print("Model ", j + 1, "had LF_K MAE: ", maei)
    MAEk.append(maei)

plt.hist(MAEz, bins=50)
plt.xlabel("Redshift Dist. MAE per test sample", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()

plt.hist(MAEk, bins=50)
plt.xlabel("LF_k MAE per test sample", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()


# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 7
for i in range(6):
    axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins[0:13], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 180
for i in range(6):
    axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:13], yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins[0:13], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 5
for i in range(6):
    axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)
    axs[i].set_xlim(-18, -25)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 180
for i in range(6):
    axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[13:22], yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)
    axs[i].set_xlim(-18, -25)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

# Plotting the MAE per input parameter, focusing on the poor predictions.
# Manual MAE score
N = len(y_test)

MAEz_filter = []
arz = []
vhdz = []
vhbz = []
ahz = []
acz = []
nsfz = []
MAEk_filter = []
ark = []
vhdk = []
vhbk = []
ahk = []
ack = []
nsfk = []
X_test = scaler_feat.inverse_transform(X_test)

for j in range(N):
    maei = MAEk[j]
    if maei > 0.1:
        # print("Model ", j + 1, "had MAE: ", maei)
        MAEk_filter.append(maei)
        ark.append(X_test[j][0])
        vhdk.append(X_test[j][1])
        vhbk.append(X_test[j][2])
        ahk.append(X_test[j][3])
        ack.append(X_test[j][4])
        nsfk.append(X_test[j][5])

for j in range(N):
    maei = MAEz[j]
    if maei > 0.1:
        # print("Model ", j + 1, "had MAE: ", maei)
        MAEz_filter.append(maei)
        arz.append(X_test[j][0])
        vhdz.append(X_test[j][1])
        vhbz.append(X_test[j][2])
        ahz.append(X_test[j][3])
        acz.append(X_test[j][4])
        nsfz.append(X_test[j][5])

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

axs[0].plot(arz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[0].plot(ark, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[0].errorbar(binar - dar / 2, medar, stdar, marker='s', color='black', alpha=0.7, label="Median")
axs[0].set_ylabel("MAE per test sample", fontsize=16)
axs[0].set_xlabel("Alpha reheat", fontsize=16)
axs[0].legend()
axs[1].plot(vhdz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[1].plot(vhdk, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[1].errorbar(binvhd - dvhd / 2, medvhd, stdvhd, marker='s', color='black', alpha=0.7, label="Median")
axs[1].set_xlabel("Vhotdisk", fontsize=16)
axs[1].legend()
axs[2].plot(vhbz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[2].plot(vhbk, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[2].errorbar(binvhb - dvhb / 2, medvhb, stdvhb, marker='s', color='black', alpha=0.7, label="Median")
axs[2].set_xlabel("Vhotbust", fontsize=16)
axs[2].legend()

axs[3].plot(ahz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[3].plot(ahk, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[3].errorbar(binah - dah / 2, medah, stdah, marker='s', color='black', alpha=0.7, label="Median")
axs[3].set_ylabel("MAE per test sample", fontsize=16)
axs[3].set_xlabel("Alpha hot", fontsize=16)
axs[3].legend()
axs[4].plot(acz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[4].plot(ack, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[4].errorbar(binac - dac / 2, medac, stdac, marker='s', color='black', alpha=0.7, label="Median")
axs[4].set_xlabel("Alpha cool", fontsize=16)
axs[4].legend()
axs[5].plot(nsfz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[5].plot(nsfk, MAEk_filter, '.', label='Luminosity Function MAE')
# axs[5].errorbar(binnsf - dnsf / 2, mednsf, stdnsf, marker='s', color='black', alpha=0.7, label="Median")
axs[5].set_xlabel("Nu SF", fontsize=16)
axs[5].legend()

plt.show()

