import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from Loading_functions import predict_all_models
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

X_test = np.load('Data/Data_for_ML/testing_data/X_test_100_fullup_int.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test_100_fullup_int.npy')

# Load all the models and make predictions on the test set
yhat_all = predict_all_models(n_models=5, X_test=X_test)
yhat_avg = np.mean(yhat_all, axis=0)

# Individual predictions for plotting
# yhat_1 = yhat_all[0]
# yhat_2 = yhat_all[1]
# yhat_3 = yhat_all[2]
# yhat_4 = yhat_all[3]
# yhat_5 = yhat_all[4]

bin_file = 'Data/Data_for_ML/bin_data/bin_fullup_int'
bins = genfromtxt(bin_file)
bins_z = bins[0:7]
bins_lfk = bins[7:25]
bins_lfr = bins[25:45]

# Manual redshift distribution MAE score
y_testz = [i[0:7] for i in y_test]
yhatz = [i[0:7] for i in yhat_avg]
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
print("\n")
# Manual luminosity function MAE score
# K-band LF
y_testk = [i[7:25] for i in y_test]
yhatk = [i[7:25] for i in yhat_avg]
# yhatk_1 = [i[13:22] for i in yhat_1]
yhatk_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk, y_testk)]
# yhatk_1_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk_1, y_testk)]
binsk = []

for i in range(100):
    bk = bins_lfk[y_testk[i] != 0]
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
print("\n")
# R-band LF
y_testr = [i[25:45] for i in y_test]
yhatr = [i[25:45] for i in yhat_avg]
# yhatk_1 = [i[13:22] for i in yhat_1]
yhatr_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatr, y_testr)]
# yhatk_1_mae = [row_a[row_b != 0] for row_a, row_b in zip(yhatk_1, y_testk)]
binsr = []

for i in range(100):
    br = bins_lfr[y_testr[i] != 0]
    binsr.append(br)
y_testr = [row[row != 0] for row in y_testr]

MAEr = []
# MAEk1 = []
for j in range(100):
    maei = mean_absolute_error(y_testr[j], yhatr_mae[j])
    # maei1 = mean_absolute_error(y_testk[j], yhatk_1_mae[j])
    if maei > 0.1:
        print("Model ", j + 1, "had LF_R MAE: ", maei)
    MAEr.append(maei)
    # MAEk1.append(maei1)

# print("\n")
# print("MAE of dn/dz for single model: ", np.mean(MAEz1))
# print("MAE of K-LF for single model: ", np.mean(MAEk1))
# print("MAE of both for single model: ", np.mean(np.vstack([MAEz1, MAEk1])))
print("\n")
print("MAE of dn/dz for average model: ", np.mean(MAEz))
print("MAE of K-LF for average model: ", np.mean(MAEk))
print("MAE of R-LF for average model: ", np.mean(MAEr))
print("MAE of all for average model: ", np.mean(np.vstack([MAEz, MAEk, MAEr])))
print("\n")

hist_bins = np.linspace(0, 0.25, 50)
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

m = 74
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

m = 85
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

m = 75
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

m = 85
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

m = 75
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

m = 85
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
# N = len(y_test)
#
# MAEz_filter = []
# arz = []
# vhdz = []
# vhbz = []
# ahz = []
# acz = []
# nsfz = []
# MAEk_filter = []
# ark = []
# vhdk = []
# vhbk = []
# ahk = []
# ack = []
# nsfk = []
# MAEr_filter = []
# arr = []
# vhdr = []
# vhbr = []
# ahr = []
# acr = []
# nsfr = []
# # X_test = scaler_feat.inverse_transform(X_test)
#
# for j in range(N):
#     maei = MAEk[j]
#     if maei > 0.0:
#         # print("Model ", j + 1, "had MAE: ", maei)
#         MAEk_filter.append(maei)
#         ark.append(X_test[j][0])
#         vhdk.append(X_test[j][1])
#         vhbk.append(X_test[j][2])
#         ahk.append(X_test[j][3])
#         ack.append(X_test[j][4])
#         nsfk.append(X_test[j][5])
#
# for j in range(N):
#     maei = MAEz[j]
#     if maei > 0.0:
#         # print("Model ", j + 1, "had MAE: ", maei)
#         MAEz_filter.append(maei)
#         arz.append(X_test[j][0])
#         vhdz.append(X_test[j][1])
#         vhbz.append(X_test[j][2])
#         ahz.append(X_test[j][3])
#         acz.append(X_test[j][4])
#         nsfz.append(X_test[j][5])
#
# for j in range(N):
#     maei = MAEr[j]
#     if maei > 0.0:
#         # print("Model ", j + 1, "had MAE: ", maei)
#         MAEr_filter.append(maei)
#         arr.append(X_test[j][0])
#         vhdr.append(X_test[j][1])
#         vhbr.append(X_test[j][2])
#         ahr.append(X_test[j][3])
#         acr.append(X_test[j][4])
#         nsfr.append(X_test[j][5])
#
# fig, axs = plt.subplots(2, 3, figsize=(15, 10),
#                         facecolor='w', edgecolor='k', sharey='row')
# fig.subplots_adjust(wspace=0)
# axs = axs.ravel()
#
# axs[0].plot(arz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[0].plot(ark, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[0].plot(arr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[0].errorbar(binar - dar / 2, medar, stdar, marker='s', color='black', alpha=0.7, label="Median")
# axs[0].set_ylabel("MAE per test sample", fontsize=16)
# axs[0].set_xlabel("Alpha reheat", fontsize=16)
# axs[0].legend()
# axs[1].plot(vhdz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[1].plot(vhdk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[1].plot(vhdr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[1].errorbar(binvhd - dvhd / 2, medvhd, stdvhd, marker='s', color='black', alpha=0.7, label="Median")
# axs[1].set_xlabel("Vhotdisk", fontsize=16)
# axs[1].legend()
# axs[2].plot(vhbz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[2].plot(vhbk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[2].plot(vhbr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[2].errorbar(binvhb - dvhb / 2, medvhb, stdvhb, marker='s', color='black', alpha=0.7, label="Median")
# axs[2].set_xlabel("Vhotbust", fontsize=16)
# axs[2].legend()
#
# axs[3].plot(ahz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[3].plot(ahk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[3].plot(ahr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[3].errorbar(binah - dah / 2, medah, stdah, marker='s', color='black', alpha=0.7, label="Median")
# axs[3].set_ylabel("MAE per test sample", fontsize=16)
# axs[3].set_xlabel("Alpha hot", fontsize=16)
# axs[3].legend()
# axs[4].plot(acz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[4].plot(ack, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[4].plot(acr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[4].errorbar(binac - dac / 2, medac, stdac, marker='s', color='black', alpha=0.7, label="Median")
# axs[4].set_xlabel("Alpha cool", fontsize=16)
# axs[4].legend()
# axs[5].plot(nsfz, MAEz_filter, '.', label='Redshift distribution MAE')
# axs[5].plot(nsfk, MAEk_filter, '.', label='K-band Luminosity Function MAE')
# axs[5].plot(nsfr, MAEr_filter, '.', label='r-band Luminosity Function MAE')
# # axs[5].errorbar(binnsf - dnsf / 2, mednsf, stdnsf, marker='s', color='black', alpha=0.7, label="Median")
# axs[5].set_xlabel("Nu SF", fontsize=16)
# axs[5].legend()
#
# plt.show()

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

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
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
axs[1].set_xlabel(r"log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] True")
axs[1].set_ylabel(r"log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] Predict")
axs[1].set_aspect('equal', 'box')
axs[1].set_xlim([-5.9, -0.3])
axs[1].set_ylim([-5.9, -0.3])
axs[2].plot(y_testr, yhatr, '.', markersize=1)
axs[2].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
axs[2].errorbar(binr - dr / 2, medr, stdr, fmt='', ecolor="black", capsize=2, alpha=0.7, linestyle='')
# axs[2].plot(binr-dr/2,running_prc25r,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
# axs[2].plot(binr-dr/2,running_prc75r,'--r',marker=None,fillstyle='none',markersize=20,alpha=1)
axs[2].set_xlabel(r"log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] True")
axs[2].set_ylabel(r"log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] Predict")
axs[2].set_aspect('equal', 'box')
axs[2].set_xlim([-5.9, -0.3])
axs[2].set_ylim([-5.9, -0.3])
plt.show()

