import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from scipy.interpolate import interp1d
import tensorflow as tf
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


# Starting best parameters
Xz_best = np.array([7.59044278e-02, 2.47684964e+02, 5.46525290e+02, 3.96208303e+00, 1.76573842e-02, 1.45159597e+00])
Xz_best = Xz_best.reshape(1, -1)
exit()

# Perform initial prediction
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
# Load in the array of models and average over the predictions
ensemble_pred = list()
for model in members:
    # Perform model prediction using the input parameters
    pred = model(Xz_best)
    ensemble_pred.append(pred)
y = np.mean(ensemble_pred, axis=0)
y = y[0]

# Plot the results

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(bins[0:49], y[0:49], 'b--', label="Best Galform predict")
axs.errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black', capsize=2,
             fmt='co', label="Bagley et al. 2020")
axs.set_xlabel('Redshift')
axs.set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs.set_xlim(0.7, 2.0)
axs.legend()
plt.show()
