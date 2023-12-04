"""
Scripts for testing the emulator against the testing datasets
This can be modified to include or not the scaling
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from Loading_functions import predict_all_models
from sklearn.preprocessing import MinMaxScaler
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)


# Load in the testing datasets
X_test = np.load('Data/Data_for_ML/testing_data/X_test_100_fullup_int_scaled.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test_100_fullup_int_scaled.npy')
# y_test = [[val] for val in y_test]

# Load all the models and make predictions on the test set
yhat_all = predict_all_models(n_models=1, X_test=X_test, variant='_9x5_scaledmask_2899_LRELU_int')
yhat_avg = np.mean(yhat_all, axis=0)

# Scaling values
nzmin = np.load('Data/Data_for_ML/min_nz_scale.npy')
nzmax = np.load('Data/Data_for_ML/max_nz_scale.npy')
kmin = np.load('Data/Data_for_ML/min_k_scale.npy')
kmax = np.load('Data/Data_for_ML/max_k_scale.npy')
rmin = np.load('Data/Data_for_ML/min_r_scale.npy')
rmax = np.load('Data/Data_for_ML/max_r_scale.npy')

# Load in the bins and split them into their statistics
bin_file = 'Data/Data_for_ML/bin_data/bin_fullup_int'
bins = genfromtxt(bin_file)
bins_z = bins[0:7]
bins_lfk = bins[7:25]
bins_lfr = bins[25:45]

# Manual redshift distribution MAE score
y_testz = [i[0:7] for i in y_test]
yhatz = [i[0:7] for i in yhat_avg]
MAEz = []
for j in range(len(y_testz)):
    # Find the MAE using the scaled values
    # y_testz_min = min(y_testz[j])
    # y_testz_max = max(y_testz[j])
    # y_test_scaled_z = (y_testz[j] - y_testz_min) / (y_testz_max - y_testz_min)
    # yhat_scaled_z = (yhatz[j] - y_testz_min) / (y_testz_max - y_testz_min)
    # maei = mean_absolute_error(y_test_scaled_z, yhat_scaled_z)
    maei = mean_absolute_error(y_testz[j], yhatz[j])
    if maei > 0.1:
        print("Model ", j + 1, "had dn/dz MAE: ", maei)
    MAEz.append(maei)
print("\n")

# Manual luminosity function MAE score
# K-band LF
y_testk = [i[7:25] for i in y_test]
yhatk = [i[7:25] for i in yhat_avg]
# Filter the datasets, so we don't calculate the MAE when K-band TRUE is zero
yhatk_mae = [row_a[row_b != -100] for row_a, row_b in zip(yhatk, y_testk)]
binsk = []
for i in range(len(y_testk)):
    bk = bins_lfk[y_testk[i] != -100]
    binsk.append(bk)
y_testk = [row[row != -100] for row in y_testk]
MAEk = []
for j in range(len(y_testk)):
    # y_testk_min = min(y_testk[j])
    # y_testk_max = max(y_testk[j])
    # y_test_scaled_k = (y_testk[j] - y_testk_min) / (y_testk_max - y_testk_min)
    # yhat_scaled_k = (yhatk_mae[j] - y_testk_min) / (y_testk_max - y_testk_min)
    # maei = mean_absolute_error(y_test_scaled_k, yhat_scaled_k)
    maei = mean_absolute_error(y_testk[j], yhatk_mae[j])
    if maei > 0.1:
        print("Model ", j + 1, "had LF_K MAE: ", maei)
    MAEk.append(maei)
print("\n")

# R-band LF
y_testr = [i[25:45] for i in y_test]
yhatr = [i[25:45] for i in yhat_avg]
# Filter the datasets, so we don't calculate the MAE when r-band TRUE is zero
yhatr_mae = [row_a[row_b != -100] for row_a, row_b in zip(yhatr, y_testr)]
binsr = []
for i in range(len(y_testr)):
    br = bins_lfr[y_testr[i] != -100]
    binsr.append(br)
y_testr = [row[row != -100] for row in y_testr]
MAEr = []
for j in range(len(y_testr)):
    # y_testr_min = min(y_testr[j])
    # y_testr_max = max(y_testr[j])
    # y_test_scaled_r = (y_testr[j] - y_testr_min) / (y_testr_max - y_testr_min)
    # yhat_scaled_r = (yhatr_mae[j] - y_testr_min) / (y_testr_max - y_testr_min)
    # maei = mean_absolute_error(y_test_scaled_r, yhat_scaled_r)
    maei = mean_absolute_error(y_testr[j], yhatr_mae[j])
    if maei > 0.1:
        print("Model ", j + 1, "had LF_R MAE: ", maei)
    MAEr.append(maei)

print("\n")
print("MAE of dn/dz for average model: ", np.mean(MAEz))
print("MAE of K-LF for average model: ", np.mean(MAEk))
print("MAE of R-LF for average model: ", np.mean(MAEr))
print("MAE of all for average model: ", np.mean(np.vstack([MAEz, MAEk, MAEr])))
print("\n")

# Inverse scale for plotting:
for i in range(len(y_test)):
    y_testz[i] = y_testz[i] * (nzmax - nzmin) + nzmin
    yhatz[i] = yhatz[i] * (nzmax - nzmin) + nzmin
    y_testk[i] = y_testk[i] * (kmax - kmin) + kmin
    yhatk[i] = yhatk[i] * (kmax - kmin) + kmin
    yhatk_mae[i] = yhatk_mae[i] * (kmax - kmin) + kmin
    y_testr[i] = y_testr[i] * (rmax - rmin) + rmin
    yhatr[i] = yhatr[i] * (rmax - rmin) + rmin
    yhatr_mae[i] = yhatr_mae[i] * (rmax - rmin) + rmin

# Create a plot that shows the histogram of MAE for each statistics
hist_bins = np.linspace(0, 0.1, 50)
plt.hist(MAEz, bins=hist_bins, label='Redshift distribution')
plt.hist(MAEk, bins=hist_bins, histtype='step', label='K-band LF')
plt.hist(MAEr, bins=hist_bins, histtype='step', label='R-band LF')
plt.xlabel("MAE per test sample")
plt.ylabel("Count")
plt.legend()
plt.show()

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

# Test model index to start the plots from
m = 59
for i in range(6):
    # axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins_z, yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins_z, y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z")

axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[3].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 28
for i in range(6):
    # axs[i].plot(bins[0:13], yhat_1[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_2[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_3[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_4[i+m][0:13], '--', alpha=0.3)
    # axs[i].plot(bins[0:13], yhat_5[i+m][0:13], '--', alpha=0.3)
    axs[i].plot(bins_z, yhatz[i+m], 'b--', label=f"Prediction MAE: {MAEz[i+m]:.3f}")
    axs[i].plot(bins_z, y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z")

axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[3].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 9
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins_lfk, yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{K,AB}$ - 5log(h)")
    axs[i].set_xlim(-15, -24)

axs[0].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
axs[3].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 67
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins_lfk, yhatk[i+m], 'b--', label=f"Prediction MAE: {MAEk[i+m]:.3f}")
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{K,AB}$ - 5log(h)")
    axs[i].set_xlim(-15, -24)

axs[0].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
axs[3].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 9
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins_lfr, yhatr[i+m], 'b--', label=f"Prediction MAE: {MAEr[i+m]:.3f}")
    axs[i].plot(binsr[i+m], y_testr[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{r,AB}$ - 5log(h)")
    axs[i].set_xlim(-13.5, -24)

axs[0].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
axs[3].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 66
for i in range(6):
    # axs[i].plot(bins[13:22], yhat_1[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_2[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_3[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_4[i+m][13:22], '--', alpha=0.3)
    # axs[i].plot(bins[13:22], yhat_5[i+m][13:22], '--', alpha=0.3)
    axs[i].plot(bins_lfr, yhatr[i+m], 'b--', label=f"Prediction MAE: {MAEr[i+m]:.3f}")
    axs[i].plot(binsr[i+m], y_testr[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("M$_{r,AB}$ - 5log(h)")
    axs[i].set_xlim(-13.5, -24)

axs[0].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
axs[3].set_ylabel('log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)')
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
Fsz = []
fez = []
fbz = []
fSz = []
tbz = []
MAEk_filter = []
ark = []
vhdk = []
vhbk = []
ahk = []
ack = []
nsfk = []
Fsk = []
fek = []
fbk = []
fSk = []
tbk = []
MAEr_filter = []
arr = []
vhdr = []
vhbr = []
ahr = []
acr = []
nsfr = []
Fsr = []
fer = []
fbr = []
fSr = []
tbr = []
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
        Fsk.append(X_test[j][6])
        fek.append(X_test[j][7])
        fbk.append(X_test[j][8])
        fSk.append(X_test[j][9])
        tbk.append(X_test[j][10])

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
        Fsz.append(X_test[j][6])
        fez.append(X_test[j][7])
        fbz.append(X_test[j][8])
        fSz.append(X_test[j][9])
        tbz.append(X_test[j][10])

for j in range(N):
    maei = MAEr[j]
    if maei > 0.0:
        # print("Model ", j + 1, "had MAE: ", maei)
        MAEr_filter.append(maei)
        arr.append(X_test[j][0])
        vhdr.append(X_test[j][1])
        vhbr.append(X_test[j][2])
        ahr.append(X_test[j][3])
        acr.append(X_test[j][4])
        nsfr.append(X_test[j][5])
        Fsr.append(X_test[j][6])
        fer.append(X_test[j][7])
        fbr.append(X_test[j][8])
        fSr.append(X_test[j][9])
        tbr.append(X_test[j][10])

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

axs[0].plot(arz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[0].plot(ark, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[0].plot(arr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[0].errorbar(binar - dar / 2, medar, stdar, marker='s', color='black', alpha=0.7, label="Median")
axs[0].set_ylabel("MAE per test sample", fontsize=16)
axs[0].set_xlabel("Alpha reheat", fontsize=16)
axs[0].legend()
axs[1].plot(vhdz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[1].plot(vhdk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[1].plot(vhdr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[1].errorbar(binvhd - dvhd / 2, medvhd, stdvhd, marker='s', color='black', alpha=0.7, label="Median")
axs[1].set_xlabel("Vhotdisk", fontsize=16)
# axs[1].legend()
axs[2].plot(vhbz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[2].plot(vhbk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[2].plot(vhbr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[2].errorbar(binvhb - dvhb / 2, medvhb, stdvhb, marker='s', color='black', alpha=0.7, label="Median")
axs[2].set_xlabel("Vhotbust", fontsize=16)
axs[2].legend()

axs[3].plot(ahz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[3].plot(ahk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[3].plot(ahr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[3].errorbar(binah - dah / 2, medah, stdah, marker='s', color='black', alpha=0.7, label="Median")
axs[3].set_ylabel("MAE per test sample", fontsize=16)
axs[3].set_xlabel("Alpha hot", fontsize=16)
# axs[3].legend()
axs[4].plot(acz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[4].plot(ack, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[4].plot(acr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[4].errorbar(binac - dac / 2, medac, stdac, marker='s', color='black', alpha=0.7, label="Median")
axs[4].set_xlabel("Alpha cool", fontsize=16)
# axs[4].legend()
axs[5].plot(nsfz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[5].plot(nsfk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[5].plot(nsfr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[5].errorbar(binnsf - dnsf / 2, mednsf, stdnsf, marker='s', color='black', alpha=0.7, label="Median")
axs[5].set_xlabel("Nu SF", fontsize=16)
# axs[5].legend()

plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

axs[0].plot(Fsz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[0].plot(Fsk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[0].plot(Fsr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[0].errorbar(binar - dar / 2, medar, stdar, marker='s', color='black', alpha=0.7, label="Median")
axs[0].set_ylabel("MAE per test sample", fontsize=16)
axs[0].set_xlabel("F stab", fontsize=16)
axs[0].legend()
axs[1].plot(fez, MAEz_filter, '.', label='Redshift distribution MAE')
axs[1].plot(fek, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[1].plot(fer, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[1].errorbar(binvhd - dvhd / 2, medvhd, stdvhd, marker='s', color='black', alpha=0.7, label="Median")
axs[1].set_xlabel("f ellip", fontsize=16)
# axs[1].legend()
axs[2].plot(fbz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[2].plot(fbk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[2].plot(fbr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[2].errorbar(binvhb - dvhb / 2, medvhb, stdvhb, marker='s', color='black', alpha=0.7, label="Median")
axs[2].set_xlabel("f burst", fontsize=16)
axs[2].legend()

axs[3].plot(fSz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[3].plot(fSk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[3].plot(fSr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[3].errorbar(binah - dah / 2, medah, stdah, marker='s', color='black', alpha=0.7, label="Median")
axs[3].set_ylabel("MAE per test sample", fontsize=16)
axs[3].set_xlabel("f SMBH", fontsize=16)
# axs[3].legend()
axs[4].plot(tbz, MAEz_filter, '.', label='Redshift distribution MAE')
axs[4].plot(tbk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
axs[4].plot(tbr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# axs[4].errorbar(binac - dac / 2, medac, stdac, marker='s', color='black', alpha=0.7, label="Median")
axs[4].set_xlabel("tau burst", fontsize=16)
# axs[4].legend()

plt.show()

# Plotting the y_pred vs y_true along the diagonal.
# Trying to fit error bars that shows the sigma range but currently can't get them to work
yhatz = np.ravel(yhatz)
y_testz = np.ravel(y_testz)
yhatk = np.concatenate(yhatk_mae).ravel().tolist()
y_testk = np.concatenate(y_testk).ravel().tolist()
yhatr = np.concatenate(yhatr_mae).ravel().tolist()
y_testr = np.concatenate(y_testr).ravel().tolist()

nbins = 50
yhatz = np.array(yhatz)
binz = np.linspace(2, max(y_testz), nbins)
dz = binz[1] - binz[0]
idxz = np.digitize(y_testz, binz)
medz = [np.median(yhatz[idxz == k]) for k in range(nbins)]
stdz = [yhatz[idxz == k].std() for k in range(nbins)]
# running25 = [np.percentile(yhatz[idxz==k], 32) for k in range(nbins)]
# running75 = [np.percentile(yhatz[idxz==k], 68) for k in range(nbins)]

yhatk = np.array(yhatk)
bink = np.linspace(-5.2, max(y_testk), nbins)
dk = bink[1] - bink[0]
idxk = np.digitize(y_testk, bink)
medk = [np.median(yhatk[idxk == k]) for k in range(nbins)]
stdk = [yhatk[idxk == k].std() for k in range(nbins)]
# running_prc25 = [np.percentile(yhatk[idxk==k], 32) for k in range(nbins)]
# running_prc75 = [np.percentile(yhatk[idxk==k], 68) for k in range(nbins)]

yhatr = np.array(yhatr)
binr = np.linspace(-5.2, max(y_testr), nbins)
dr = binr[1] - binr[0]
idxr = np.digitize(y_testr, binr)
medr = [np.median(yhatr[idxr == k]) for k in range(nbins)]
stdr = [yhatr[idxr == k].std() for k in range(nbins)]
# running_prc25r = [np.percentile(yhatk[idxr==k], 32) for k in range(nbins)]
# running_prc75r = [np.percentile(yhatk[idxr==k], 68) for k in range(nbins)]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(y_testz, yhatz, '.', markersize=1)
axs[0].axline((2, 2), slope=1, color='black', linestyle='dotted')
axs[0].errorbar(binz - dz / 2, medz, stdz, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
# axs[0].plot(binz-dz/2,running25,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
# axs[0].plot(binz-dz/2,running75,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[0].set_xlabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$] True")
axs[0].set_ylabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$] Predict")
axs[0].set_aspect('equal', 'box')
axs[0].set_xlim([1.6, 4.6])
axs[0].set_ylim([1.6, 4.6])
axs[1].plot(y_testk, yhatk, '.', markersize=1)
axs[1].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
axs[1].errorbar(bink - dk / 2, medk, stdk, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
# axs[1].plot(bink-dk/2,running_prc25,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
# axs[1].plot(bink-dk/2,running_prc75,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[1].set_xlabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{k,AB}$)$^{-1}$) True")
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{k,AB}$)$^{-1}$) Predict")
axs[1].set_aspect('equal', 'box')
axs[1].set_xlim([-5.9, -0.3])
axs[1].set_ylim([-5.9, -0.3])
axs[2].plot(y_testr, yhatr, '.', markersize=1)
axs[2].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
axs[2].errorbar(binr - dr / 2, medr, stdr, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
# axs[2].plot(binr-dr/2,running_prc25r,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
# axs[2].plot(binr-dr/2,running_prc75r,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[2].set_xlabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{r,AB}$)$^{-1}$) True")
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{r,AB}$)$^{-1}$) Predict")
axs[2].set_aspect('equal', 'box')
axs[2].set_xlim([-5.9, -0.3])
axs[2].set_ylim([-5.9, -0.3])
plt.show()

