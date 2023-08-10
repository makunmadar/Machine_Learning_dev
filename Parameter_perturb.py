import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from Loading_functions import kband_df, load_all_models

# Import the observational data for reference
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]
sigma = (Ha_ytop + Ha_ybot) / 2

# Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = kband_df(drive_path, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

# Load in the Galform bins
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

# Perform initial prediction
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))

param_range = [3.0, 450.0, 450.0, 2.5, 2.0, 1.5]
min_bound = np.array([0.0, 100, 100, 1.5, 0.0, 0.2])
max_bound = np.array([3.0, 550, 550, 4.0, 2.0, 1.7])
b = [i / 40 for i in param_range]

figz, axz = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey=True)
axz = axz.ravel()

figk, axk = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k', sharey=True)
axk = axk.ravel()

labels = [r"$\alpha_{ret}$", r"$V_{SN, disk}$", r"$V_{SN, burst}$",
          r"$\gamma_{SN}$", r"$\alpha_{cool}$", r"$\nu_{SF}$ [Gyr$^{-1}$]"]

for i in range(6):
    # Reset the parameters:
    Xz_best = np.array([2.86048669, 350.5901748, 539.77983928, 3.94626232, 0.98279349, 1.50068536])
    Xz_best = Xz_best.reshape(1, -1)

    while np.all(Xz_best < max_bound):
        # Load in the array of models and average over the predictions
        ensemble_pred = list()
        for model in members:
            # Perform model prediction using the input parameters
            pred = model(Xz_best)
            ensemble_pred.append(pred)
        predictions = np.mean(ensemble_pred, axis=0)

        # Perform model prediction using the input parameters
        y = predictions[0]

        axz[i].plot(bins[0:49], y[0:49], c='red', alpha=0.5)
        axk[i].plot(bins[49:67], y[49:67], c='red', alpha=0.5)
        Xz_best[0][i] += b[i]

    Xz_best = np.array([2.86048669, 350.5901748, 539.77983928, 3.94626232, 0.98279349, 1.50068536])
    Xz_best = Xz_best.reshape(1, -1)

    while np.all(Xz_best > min_bound):
        # Load in the array of models and average over the predictions
        ensemble_pred = list()
        for model in members:
            # Perform model prediction using the input parameters
            pred = model(Xz_best)
            ensemble_pred.append(pred)
        predictions = np.mean(ensemble_pred, axis=0)

        # Perform model prediction using the input parameters
        y = predictions[0]

        axz[i].plot(bins[0:49], y[0:49], c='blue', alpha=0.5)
        axk[i].plot(bins[49:67], y[49:67], c='blue', alpha=0.5)
        Xz_best[0][i] -= b[i]

    axz[i].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black', capsize=2,
                      fmt='co', label="Bagley et al. 2020")
    axz[i].set_xlabel('Redshift', fontsize=10)
    axz[i].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=10)
    axz[i].set_xlim(0.7, 2.0)
    axz[i].legend()
    axz[i].set_title(labels[i])

    axk[i].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                      markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
    axk[i].legend()
    axk[i].set_xlabel(r"M$_{AB}$ - 5log(h)", fontsize=10)
    axk[i].set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)", fontsize=10)
    axk[i].set_xlim(-18, -25)
    axk[i].set_ylim(-6, -1)
    axk[i].set_title(labels[i])

plt.show()