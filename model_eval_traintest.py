import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error

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
bins_k = bins[13:22]

# Manual redshift distribution MAE score
y_testz = [i[0:13] for i in y_test]
yhatz = [i[0:13] for i in yhat]

MAEz = []
for j in range(200):
    maei = mean_absolute_error(y_testz[j], yhatz[j])
    if maei > 0.2:
        print("Model ", j + 1, "had dn/dz MAE: ", maei)
    MAEz.append(maei)

# Manual luminosity function MAE score
y_testk = [i[13:22] for i in y_test]
yhatk = [i[13:22] for i in yhat]

yhatk = [row_a[row_b != 0] for row_a, row_b in zip(yhatk, y_testk)]
binsk = []
for i in range(200):
    bk = bins_k[y_testk[i] != 0]
    binsk.append(bk)
y_testk = [row[row != 0] for row in y_testk]

MAEk = []
for j in range(200):
    maei = mean_absolute_error(y_testk[j], yhatk[j])
    if maei > 0.7:
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

m = 138
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

m = 180
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

m = 152
for i in range(6):
    axs[i].plot(binsk[i+m], yhatk[i+m], '--', label=f"Prediction MAE: {MAEk[i+m]:.3f}", alpha=0.5)
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 165
for i in range(6):
    axs[i].plot(binsk[i+m], yhatk[i+m], '--', label=f"Prediction MAE: {MAEk[i+m]:.3f}", alpha=0.5)
    axs[i].plot(binsk[i+m], y_testk[i+m], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()
