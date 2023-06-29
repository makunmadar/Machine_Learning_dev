import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import dump, load

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

# Manual MAE score
n = len(bins[0:13])
N = len(y_test)

MAE = []
for j in range(N):
    maei = 0
    for i in range(n):
        sumi = abs(y_test[j][i] - yhat[j][i])
        maei += sumi
    maei = maei / n
    if maei > 0.3:
        print("Model ", j + 1, "had MAE: ", maei)
    MAE.append(maei)

plt.hist(MAE, bins=50)
plt.xlabel("Total MAE per test sample", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey='row')
fig.subplots_adjust(wspace=0)
axs = axs.ravel()

m = 66
for i in range(6):
    axs[i].plot(bins[0:13], yhat[i+m][0:13], '--', label=f"Prediction MAE: {MAE[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[0:13], y_test[i+m][0:13], 'gx-', label="True model "+str(i+1+m))
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
    axs[i].plot(bins[0:13], yhat[i+m][0:13], '--', label=f"Prediction MAE: {MAE[i+m]:.3f}", alpha=0.5)
    axs[i].plot(bins[0:13], y_test[i+m][0:13], 'gx-', label="True model "+str(i+1+m))
    axs[i].legend()
    axs[i].set_xlabel("Redshift, z", fontsize=16)

axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)
plt.show()