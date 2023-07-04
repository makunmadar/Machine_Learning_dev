import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load

X_test = np.load('Data/Data_for_ML/testing_data/X_test.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test.npy')

scaler_feat = load('mm_scaler_feat.bin')
X_test = scaler_feat.transform(X_test)
scaler_label = load('std_scaler_label.bin')

model = tf.keras.models.load_model('Models/Ensemble_model_1_1000_S', compile=False)

yhat = model.predict(X_test)
yhat = scaler_label.inverse_transform(yhat)

bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

# Manual redshift distribution MAE score
nz = len(bins[0:13])
Nz = len([i[0:13] for i in y_test])
y_testz = [i[0:13] for i in y_test]
yhatz = [i[0:13] for i in yhat]

MAEz = []
for j in range(Nz):
    maei = 0
    for i in range(nz):
        sumi = abs(y_testz[j][i] - yhatz[j][i])
        maei += sumi
    maei = maei / nz
    if maei > 0.3:
        print("Model ", j + 1, "had dn/dz MAE: ", maei)
    MAEz.append(maei)

# Manual luminosity function MAE score

nk = len(bins[13:22])
Nk = len([i[13:22] for i in y_test])
y_testk = [i[13:22] for i in y_test]
yhatk = [i[13:22] for i in yhat]

MAEk = []
for j in range(Nk):
    maei = 0
    for i in range(nk):
        sumi = abs(y_testk[j][i] - yhatk[j][i])
        maei += sumi
    maei = maei / nk
    if maei > 0.3:
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

m = 66
for i in range(6):
    axs[i].plot(bins[0:13], yhatz[i+m], '--', label=f"Prediction MAE: {MAEz[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[0:13], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 182
for i in range(6):
    axs[i].plot(bins[0:13], yhatz[i+m], '--', label=f"Prediction MAE: {MAEz[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[0:13], y_testz[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 66
for i in range(6):
    axs[i].plot(bins[13:22], yhatk[i+m], '--', label=f"Prediction MAE: {MAEk[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[13:22], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 182
for i in range(6):
    axs[i].plot(bins[13:22], yhatk[i+m], '--', label=f"Prediction MAE: {MAEk[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[13:22], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()
