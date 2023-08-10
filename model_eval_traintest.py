import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from Loading_functions import predict_all_models


X_test = np.load('Data/Data_for_ML/testing_data/X_test_100_full.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test_100_full.npy')

# scaler_feat = load('mm_scaler_feat_900_full.bin')
# X_test = scaler_feat.transform(X_test)
# scaler_label = load('std_scaler_label.bin')

# Load all the models and make predictions on the test set
yhat_all = predict_all_models(n_models=5, X_test=X_test)
yhat_avg = np.mean(yhat_all, axis=0)

# Individual predictions for plotting
# yhat_1 = yhat_all[0]
# yhat_2 = yhat_all[1]
# yhat_3 = yhat_all[2]
# yhat_4 = yhat_all[3]
# yhat_5 = yhat_all[4]

bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)
bins_k = bins[49:67]

# Manual redshift distribution MAE score
y_testz = [i[0:49] for i in y_test]
yhatz = [i[0:49] for i in yhat_avg]
# yhatz_1 = [i[0:13] for i in yhat_1]

MAEz = []
# MAEz1 = []
for j in range(100):
    maei = mean_absolute_error(y_testz[j], yhatz[j])
    # maei1 = mean_absolute_error(y_testz[j], yhatz_1[j])
    if maei > 0.1:
        print("Model ", j + 1, "had dn/dz MAE: ", maei)
    MAEz.append(maei)
    # MAEz1.append(maei1)

# Manual luminosity function MAE score
y_testk = [i[49:67] for i in y_test]
yhatk = [i[49:67] for i in yhat_avg]
# yhatk_1 = [i[13:22] for i in yhat_1]
yhatk_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk, y_testk)]
# yhatk_1_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk_1, y_testk)]
binsk = []

for i in range(100):
    bk = bins_k[y_testk[i] != 0]
    binsk.append(bk)
y_testk = [row[row != 0] for row in y_testk]

MAEk = []
# MAEk1 = []
for j in range(100):
    maei = mean_absolute_error(y_testk[j], yhatk_mae[j])
    # maei1 = mean_absolute_error(y_testk[j], yhatk_1_mae[j])
    if maei > 0.1:
        print("Model ", j + 1, "had LF_K MAE: ", maei)
    MAEk.append(maei)
    # MAEk1.append(maei1)

# print("\n")
# print("MAE of dn/dz for single model: ", np.mean(MAEz1))
# print("MAE of K-LF for single model: ", np.mean(MAEk1))
# print("MAE of both for single model: ", np.mean(np.vstack([MAEz1, MAEk1])))
print("\n")
print("MAE of dn/dz for average model: ", np.mean(MAEz))
print("MAE of K-LF for average model: ", np.mean(MAEk))
print("MAE of both for average model: ", np.mean(np.vstack([MAEz, MAEk])))
print("\n")

plt.hist(MAEz, bins=50, label='Redshift distribution')
plt.hist(MAEk, bins=50, histtype='step', label='K-band LF')
plt.xlabel("MAE per test sample", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.legend()
plt.show()

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 30
for i in range(6):
    # axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:49], yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins[0:49], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 50
for i in range(6):
    # axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins[0:49], yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins[0:49], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
axs[3].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 20
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[49:67], yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{AB}$ - 5log(h)", fontsize=16)
    axs[i].set_xlim(-18, -25)

axs[0].set_ylabel('Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)', fontsize=16)
axs[3].set_ylabel('Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 50
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins[49:67], yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{AB}$ - 5log(h)", fontsize=16)
    axs[i].set_xlim(-18, -25)

axs[0].set_ylabel('Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)', fontsize=16)
axs[3].set_ylabel('Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)', fontsize=16)
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
# X_test = scaler_feat.inverse_transform(X_test)

for j in range(N):
    maei = MAEk[j]
    if maei > 0.0:
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
    if maei > 0.0:
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

yhatz = np.ravel(yhatz)
y_testz = np.ravel(y_testz)
yhatk = np.concatenate(yhatk_mae).ravel().tolist()
y_testk = np.concatenate(y_testk).ravel().tolist()

nbins = 50
yhatz = np.array(yhatz)
binz = np.linspace(2, max(y_testz), nbins)
dz = binz[1] - binz[0]
idxz = np.digitize(y_testz, binz)
medz = [np.median(yhatz[idxz == k]) for k in range(nbins)]
stdz = [yhatz[idxz == k].std() for k in range(nbins)]
running25 = [np.percentile(yhatz[idxz==k], 32) for k in range(nbins)]
running75 = [np.percentile(yhatz[idxz==k], 68) for k in range(nbins)]

yhatk = np.array(yhatk)
bink = np.linspace(-5.2, max(y_testk), nbins)
dk = bink[1] - bink[0]
idxk = np.digitize(y_testk, bink)
medk = [np.median(yhatk[idxk == k]) for k in range(nbins)]
stdk = [yhatk[idxk == k].std() for k in range(nbins)]
running_prc25 = [np.percentile(yhatk[idxk==k], 32) for k in range(nbins)]
running_prc75 = [np.percentile(yhatk[idxk==k], 68) for k in range(nbins)]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(y_testz, yhatz, '.', markersize=1)
axs[0].axline((2, 2), slope=1, color='black', linestyle='dotted')
axs[0].errorbar(binz - dz / 2, medz, stdz, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
axs[0].plot(binz-dz/2,running25,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[0].plot(binz-dz/2,running75,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[0].set_xlabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$] True", fontsize=15)
axs[0].set_ylabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$] Predict", fontsize=15)
axs[0].set_aspect('equal', 'box')
axs[0].set_xlim([1.6, 4.6])
axs[0].set_ylim([1.6, 4.6])
axs[1].plot(y_testk, yhatk, '.', markersize=1)
axs[1].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
axs[1].errorbar(bink - dk / 2, medk, stdk, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
axs[1].plot(bink-dk/2,running_prc25,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[1].plot(bink-dk/2,running_prc75,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[1].set_xlabel(r"Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] True", fontsize=15)
axs[1].set_ylabel(r"Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] Predict", fontsize=15)
axs[1].set_aspect('equal', 'box')
axs[1].set_xlim([-5.9, -0.3])
axs[1].set_ylim([-5.9, -0.3])
plt.show()

