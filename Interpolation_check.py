"""Testing the interpolation code between the galform predictions and the observables for MAE calculations"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
from Loading_functions import lf_df, load_all_models
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# Import the Bagley et al. 2020 data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
sigmaz = (Ha_b["+"].values - Ha_b['-'].values) / 2

Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]

# Import the Driver et al. 2012 data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path_k = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_path_k, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
sigmak = df_k['error'].values

df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

drive_path_r = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_path_r, driv_headers)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
sigmar = df_r['error'].values

df_r['error_upper'] = np.log10(df_r['LF'] + df_r['error']) - np.log10(df_r['LF'])
df_r['error_lower'] = np.log10(df_r['LF']) - np.log10(df_r['LF'] - df_r['error'])
df_r['LF'] = np.log10(df_r['LF'])

# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)
# Import Lacey et al. 2016 for comparison
Lacey_y = np.load("Lacey_y_true.npy")

# Test with a random prediction
# An example of where the models thinks there is actually an increase in the LF at the bright end.
X_rand = np.array([1.35075880e-03, 2.45363711e+02, 1.06058426e+02, 3.37190159e+00, 6.78036847e-01, 5.74418884e-01])
# Lacey et al. 2016
# X_rand = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
# X_rand = np.array([2.97606331, 371.77262632, 104.98390629, 3.97517725, 0.88732652, 1.64713981])
X_rand = X_rand.reshape(1, -1)

members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
# Load in the array of models and average over the predictions
y = np.mean([model(X_rand) for model in members], axis=0)

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

# Try on the Driver et al. 2012 K-band LF data
# Perform interpolation
xk1 = bins[49:67]
yk1 = y[49:67]
xk2 = df_k['Mag'].values
yk2 = df_k['LF'].values

# Interpolate or resample the data onto the common x-axis
interp_funck = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
interp_yk1 = interp_funck(xk2)

weighted_maek = mean_absolute_error(yk2, interp_yk1)
print("MAE K-band luminosity function: ", weighted_maek)

# Try on the Driver et al. 2012 r-band LF data
# Perform interpolation
xr1 = bins[49:67]
yr1 = y[67:85]
xr2 = df_r['Mag'].values
yr2 = df_r['LF'].values

# Interpolate or resample the data onto the common x-axis
interp_funcr = interp1d(xr1, yr1, kind='linear', fill_value='extrapolate')
interp_yr1 = interp_funcr(xr2)

weighted_maer = mean_absolute_error(yr2, interp_yr1)
print("MAE r-band luminosity function: ", weighted_maer)

# Plot to see how this looks
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(bins[0:49], y[0:49], 'b--', label="Galform prediction")
axs[0].plot(bins[0:49], Lacey_y[0:49], 'r--', label="Lacey et al. 2016")
# axs[0].plot(xz2, interp_yz1, 'bx', label='Interpolated galform')
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[0].set_xlim(0.7, 2.0)
axs[0].legend()
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
axs[1].plot(bins[49:67], y[49:67], 'b--', label='Galform prediction')
axs[1].plot(bins[51:67], Lacey_y[49:65], 'r--', label="Lacey et al. 2016")
# axs[1].plot(xk2, interp_yk1, 'bx', label='Interpolated galform')
axs[1].set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{AB}$ - 5log(h)")
axs[1].set_xlim(-18, -25)
axs[1].set_ylim(-6, -1)
axs[1].legend()
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
axs[2].plot(bins[49:67], y[67:85], 'b--', label='Galform prediction')
axs[2].plot(bins[53:67], Lacey_y[65:79], 'r--', label="Lacey et al. 2016")
# axs[2].plot(xr2, interp_yr1, 'bx', label='Interpolated galform')
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-18, -25)
axs[2].set_ylim(-6, -1)
axs[2].legend()
plt.show()

# Working out the MAE values using Lagrangian likelihood:
pred = 10 ** (np.hstack([interp_yz1, interp_yk1, interp_yr1]))
obs = 10 ** (np.hstack([yz2, yk2, yr2]))
print(obs)
sigma = np.hstack([sigmaz, sigmak, sigmar])
# # Need to apply scaling:
# min_value = np.min([np.min(pred), np.min(obs)])
# max_value = np.max([np.max(pred), np.max(obs)])
# scaled_pred = (pred - min_value) / (max_value - min_value)
# scaled_obs = (obs - min_value) / (max_value - min_value)

# Manually calculate the weighted MAE
abs_diff = np.abs(obs - pred) / sigma
fract = sigma/obs
print("Fractional error: ", fract)
print("\n")
# print("Abs diff: ", abs_diff)
# abs_diff = ((pred-obs)**2)/sigma**2

# bag_i = abs_diff[0:7] / 7
# driv_i = abs_diff[7:19] / 12
# mae = (1 / 2) * (np.sum(bag_i) + np.sum(driv_i))
mae = np.mean(abs_diff)
print("Unweighted MAE: ", mae)
print("Redshift distribution MAE: ", np.mean(abs_diff[0:7]))
print("K-band Luminosity function MAE: ", np.mean(abs_diff[7:19]))
print("r-band Luminosity function MAE: ", np.mean(abs_diff[19:30]))

# Convert to Lagrangian likelihood
# likelihood = (1/(2*0.2)) * np.exp(-mae / 0.2)
# likelihood = np.exp(-weighted_mae / 2)
# print("Likelihood (b=0.2): ", likelihood)

# Testing Laplacian distibution
# initial_ar = np.random.uniform(0.3, 3.0)
# initial_vd = np.random.uniform(100, 550)
# initial_vb = np.random.uniform(100, 550)
# initial_ah = np.random.uniform(1.5, 3.5)
# initial_ac = np.random.uniform(0.0, 2.0)
# initial_ns = np.random.uniform(0.2, 1.7)
# initial_state = np.array([initial_ar, initial_vd, initial_vb, initial_ah, initial_ac, initial_ns])
# initial_state = initial_state.reshape(1, -1)
# print("Initial random state: ", initial_state)
#
# # Scale parameter is 1/20th the parameter range
# # same as the original step size.
# param_range = [2.7, 450.0, 450.0, 2.0, 2.0, 1.5]
# b = [i/100 for i in param_range]
# L = np.random.laplace(scale=b)
# print("Lagrangian values: ", L)
#
# new_state = initial_state + L
# print("Proposed new state: ", new_state)

# Commutative plot
LF_error_k = abs_diff[7:19]
cum_error_k = np.cumsum(LF_error_k)
LF_error_r = abs_diff[19:30]
cum_error_r = np.cumsum(LF_error_r)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
color = "tab:blue"
axs[0].errorbar(df_k["Mag"], df_k["LF"], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='black',
             ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
l2k = axs[0].plot(bins[49:67], y[49:67], '--', color=color, label='Galform prediction')
axs[0].tick_params(axis='y', labelcolor=color)
axs[0].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[0].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].errorbar(df_r["Mag"], df_r["LF"], yerr=(df_r['error_lower'], df_r['error_upper']), markeredgecolor='black',
             ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
l2r = axs[1].plot(bins[49:67], y[67:85], '--', color=color, label='Galform prediction')
axs[1].tick_params(axis='y', labelcolor=color)
axs[1].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")

axsk = axs[0].twinx()
axsr = axs[1].twinx()
color = "tab:red"
l3k = axsk.plot(xk2, cum_error_k, color=color, label="Cumulative error")
l3r = axsr.plot(xr2, cum_error_r, color=color, label="Cumulative error")
axsk.tick_params(axis='y', labelcolor=color)
axsr.tick_params(axis='y', labelcolor=color)
axsk.set_ylabel("Cumulative error (MAE)")
axsr.set_ylabel("Cumulative error (MAE)")
legk = l2k + l3k
labsk = [l.get_label() for l in legk]
axs[0].legend(legk, labsk, loc=0)
legr = l2r + l3r
labsr = [l.get_label() for l in legr]
axs[1].legend(legr, labsr, loc=0)
fig.tight_layout()
plt.show()
