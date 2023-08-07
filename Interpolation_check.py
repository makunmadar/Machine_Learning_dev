"""Testing the interpolation code between the galform predictions and the observables for MAE calculations"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import tensorflow as tf
from joblib import load
from sklearn.metrics import mean_absolute_error
from Loading_functions import kband_df, load_all_models


# Import the Bagley et al. 2020 data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]
sigmaz = (Ha_ytop + Ha_ybot) / 2

# Import one example from the testing set:
# X_all = np.load('Data/Data_for_ML/testing_data/X_test_200.npy')
# y_all = np.load('Data/Data_for_ML/testing_data/y_test_200.npy')
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
# X = X_all[1]
# y = y_all[1]
bins = genfromtxt(bin_file)

# Test with a random prediction
# An example of where the models thinks there is actually an increase in the LF at the bright end.
# X_rand = np.array([2.33, 545.51, 227.26, 2.93, 0.69, 0.59])
# Lacey et al. 2016
# X_rand = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
X_rand = np.array([2.86048669, 350.5901748, 539.77983928, 3.94626232, 0.98279349, 1.50068536])
# X_rand = np.array([2.51037146e-02, 1.78693367e+02, 2.46481835e+02, 3.9979414, 7.84771343e-01, 8.19003567e-01])
X_rand = X_rand.reshape(1, -1)

# scaler_feat = load("mm_scaler_feat_900_full.bin")
# X_rand = scaler_feat.transform(X_rand)

# model = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask',
#                                    custom_objects={"masked_mae": masked_mae}, compile=False)
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
# Load in the array of models and average over the predictions
ensemble_pred = list()
for model in members:
    # Perform model prediction using the input parameters
    pred = model(X_rand)
    ensemble_pred.append(pred)
y = np.mean(ensemble_pred, axis=0)


# y = model.predict(X_rand)
y = y[0]

# Redshift distribution
# Perform interpolation
xz1 = bins[0:49]
yz1 = y[0:49]
xz2 = Ha_b['z'].values
yz2 = Ha_b['n'].values

# Interpolate or resample the daa onto the common x-axis
interp_funcz = interp1d(xz1, yz1, kind='linear', fill_value='extrapolate')
interp_yz1 = interp_funcz(xz2)

# Working out the MAE values
weighted_maez = mean_absolute_error(yz2, interp_yz1)
print("MAE redshift distribution: ", weighted_maez)

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

# Perform interpolation
xk1 = bins[49:67]
yk1 = y[49:67]
xk2 = df_k['Mag'].values
yk2 = df_k['LF'].values

# Interpolate or resample the data onto the common x-axis
interp_funck = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
interp_yk1 = interp_funck(xk2)

# Plot to see how this looks
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(bins[0:49], y[0:49], 'b--', label="Galform prediction")
axs[0].plot(xz2, interp_yz1, 'bx', label='Interpolated galform')
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
             fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlim(0.7, 2.0)
axs[0].legend()
df_k.plot(ax=axs[1], x="Mag", y="LF", label='Driver et al. 2012', yerr=[df_k['error_lower'], df_k['error_upper']],
          markeredgecolor='black', ecolor="black", capsize=2, fmt='co')
axs[1].plot(bins[49:67], y[49:67], 'b--', label='Galform prediction')
axs[1].plot(xk2, interp_yk1, 'bx', label='Interpolated galform')
axs[1].set_xlim(-18, -25)
axs[1].set_ylim(-6, -1)
axs[1].legend()
plt.show()

weighted_maek = mean_absolute_error(yk2, interp_yk1)
print("MAE luminosity function: ", weighted_maek)

# Working out the MAE values using Lagrangian likelihood:
mae_weighting = [1.0] * 7 + [0.5] * 12
pred = np.hstack([interp_yz1, interp_yk1])
obs = np.hstack([yz2, yk2])

# # Need to apply scaling:
min_value = np.min([np.min(pred), np.min(obs)])
max_value = np.max([np.max(pred), np.max(obs)])
scaled_pred = (pred - min_value) / (max_value - min_value)
scaled_obs = (obs - min_value) / (max_value - min_value)

# Manually calculate the weighted MAE
abs_diff = np.abs(scaled_pred - scaled_obs)
weighted_diff = mae_weighting * abs_diff

bag_i = weighted_diff[0:7] / 7
driv_i = weighted_diff[7:19] / 12

weighted_mae = (1 / 2) * (np.sum(bag_i) + np.sum(driv_i))
print("Weighted MAE: ", weighted_mae)

# Convert to Lagrangian likelihood
likelihood = np.prod((1 / (2 * 0.05)) * np.exp(-weighted_mae / 0.05))
print("Likelihood: ", likelihood)

# Testing Laplacian distibution
initial_ar = np.random.uniform(0.3, 3.0)
initial_vd = np.random.uniform(100, 500)
initial_vb = np.random.uniform(100, 500)
initial_ah = np.random.uniform(1.5, 3.5)
initial_ac = np.random.uniform(0.0, 2.0)
initial_ns = np.random.uniform(0.2, 1.7)
initial_state = np.array([initial_ar, initial_vd, initial_vb, initial_ah, initial_ac, initial_ns])
initial_state = initial_state.reshape(1, -1)
print("Initial random state: ", initial_state)

# Scale parameter is 1/20th the parameter range
# same as the original step size.
param_range = [2.7, 450.0, 450.0, 2.0, 2.0, 1.5]
b = [i/20 for i in param_range]
L = np.random.laplace(scale=b)
print("Lagrangian values: ", L)

new_state = initial_state + L
print("Proposed new state: ", new_state)
