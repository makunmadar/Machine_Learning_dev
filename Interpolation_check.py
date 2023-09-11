"""Testing the interpolation code between the galform predictions and the observables for MAE calculations"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
from Loading_functions import lf_df, load_all_models
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# Import the Bagley et al. 2020 data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
# sigmaz = (Ha_b["+"].values - Ha_b['-'].values) / 2

Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]

# Import the Driver et al. 2012 data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path_k = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_path_k, driv_headers, mag_high=-15.25, mag_low=-23.75)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
# sigmak = df_k['error'].values

df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

drive_path_r = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_path_r, driv_headers, mag_high=-13.75, mag_low=-23.25)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
# sigmar = df_r['error'].values

df_r['error_upper'] = np.log10(df_r['LF'] + df_r['error']) - np.log10(df_r['LF'])
df_r['error_lower'] = np.log10(df_r['LF']) - np.log10(df_r['LF'] - df_r['error'])
df_r['LF'] = np.log10(df_r['LF'])

# Import Cole et al. 2001
cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
cole_path_k = 'Data/Data_for_ML/Observational/Cole_2001/lfJK_Cole2001.data'
df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00, mag_high=-18.00)
df_ck = df_ck[df_ck['PhiK'] != 0]
df_ck = df_ck.sort_values(['Mag'], ascending=[True])
df_ck['errorK_upper'] = np.log10(df_ck['PhiK'] + df_ck['errorK']) - np.log10(df_ck['PhiK'])
df_ck['errorK_lower'] = np.log10(df_ck['PhiK']) - np.log10(df_ck['PhiK'] - df_ck['errorK'])
df_ck['PhiK'] = np.log10(df_ck['PhiK'])

# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full_int_colek'
bins = genfromtxt(bin_file)
# Import Lacey et al. 2016 for comparison
Lacey_y = np.load("Lacey_y_true.npy")
bins_l = np.load('Lacey_bins.npy')

# Test with a random prediction
# An example of where the models thinks there is actually an increase in the LF at the bright end.
# The following parameters are what we are checking against using the scaling method
# X_rand_old = np.array([1.80595566e-01, 3.13150522e+02, 4.82625973e+02, 3.46543453e+00, 2.94445793e-05, 1.67638043e+00])
# ratio_old = "4:1:1"
# Best fitting non scaling method
X_rand = np.array([2.24849824e+00, 4.72013031e+02, 1.06094650e+02, 3.49803642e+00, 1.88328839e-01, 3.45516933e-01])
ratio = "1:1:1"
X_rand_old = np.array([4.87305037e-02, 2.31174804e+02, 1.16825859e+02, 3.49499532e+00, 3.45075922e-01, 1.66716602e+00])
ratio_old = "3.5:1:1"
# Lacey et al. 2016
# X_rand = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
X_rand = X_rand.reshape(1, -1)

members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
# Load in the array of models and average over the predictions
y = np.mean([model(X_rand) for model in members], axis=0)
y_old = np.mean([model(X_rand_old) for model in members], axis=0)

y = y[0]
y_old = y_old[0]

# Redshift distribution
# Perform interpolation
yz1 = y[0:7]
xz2 = Ha_b['z'].values
yz2 = Ha_b['n'].values

# Interpolate or resample the daa onto the common x-axis
# interp_funcz = interp1d(xz1, yz1, kind='linear', fill_value='extrapolate')
# interp_yz1 = interp_funcz(xz2)

# Working out the MAE values
weighted_maez = mean_absolute_error(yz2, yz1)
print("MAE redshift distribution: ", weighted_maez)

# Try on the Driver et al. 2012 K-band LF data
# Perform interpolation
yk1 = y[7:31]
xk2 = df_ck['Mag'].values
yk2 = df_ck['PhiK'].values

# Interpolate or resample the data onto the common x-axis
# interp_funck = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
# interp_yk1 = interp_funck(xk2)

weighted_maek = mean_absolute_error(yk2, yk1)
print("MAE K-band luminosity function: ", weighted_maek)

# Try on the Driver et al. 2012 r-band LF data
# Perform interpolation
yr1 = y[31:51]
xr2 = df_r['Mag'].values
yr2 = df_r['LF'].values

# Interpolate or resample the data onto the common x-axis
# interp_funcr = interp1d(xr1, yr1, kind='linear', fill_value='extrapolate')
# interp_yr1 = interp_funcr(xr2)

weighted_maer = mean_absolute_error(yr2, yr1)
print("MAE r-band luminosity function: ", weighted_maer)

# Plot to see how this looks
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(bins[0:7], yz1, 'b--', label="Galform prediction " + ratio)
axs[0].plot(bins_l[0:49], Lacey_y[0:49], 'r--', label="Lacey et al. 2016")
# axs[0].plot(xz2, interp_yz1, 'bx', label='Interpolated galform')
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[0].set_xlim(0.9, 1.6)
axs[0].legend()
# axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#                 markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
axs[1].errorbar(df_ck['Mag'], df_ck['PhiK'], yerr=(df_ck['errorK_lower'], df_ck['errorK_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Cole et al. 2001')
axs[1].plot(bins[7:31], yk1, 'b--', label='Galform prediction ' + ratio)
axs[1].plot(bins_l[49:69], Lacey_y[49:69], 'r--', label="Lacey et al. 2016")
# axs[1].plot(xk2, interp_yk1, 'bx', label='Interpolated galform')
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{AB}$ - 5log(h)")
axs[1].set_xlim(-18, -24)
axs[1].set_ylim(-6, -1)
axs[1].legend()
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].plot(bins[31:51], yr1, 'b--', label='Galform prediction ' + ratio)
axs[2].plot(bins_l[69:91], Lacey_y[69:91], 'r--', label="Lacey et al. 2016")
# axs[2].plot(xr2, interp_yr1, 'bx', label='Interpolated galform')
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].legend()
plt.show()

# Working out the MAE values using Lagrangian likelihood:
# pred = np.hstack([yz1, yk1, yr1])
obs = np.hstack([yz2, yk2, yr2])
# sigma = np.hstack([sigmaz, sigmak, sigmar])
# # Need to apply scaling:
# min_zvalue = np.min(obs[0:7])
# max_zvalue = np.max(obs[0:7])
# scaling_factorz = 1.0 / (max_zvalue - min_zvalue)
# offsetz = -min_zvalue * scaling_factorz
# scaled_yz1 = (pred[0:7] * scaling_factorz) + offsetz
# scaled_yz2 = (obs[0:7] * scaling_factorz) + offsetz

# min_kvalue = np.min(obs[7:25])
# max_kvalue = np.max(obs[7:25])
# scaling_factork = 1.0 / (max_kvalue - min_kvalue)
# offsetk = -min_kvalue * scaling_factork
# scaled_yk1= (pred[7:25] * scaling_factork) + offsetk
# scaled_yk2 = (obs[7:25] * scaling_factork) + offsetk

# min_rvalue = np.min(obs[25:45])
# max_rvalue = np.max(obs[25:45])
# scaling_factorr = 1.0 / (max_rvalue - min_rvalue)
# offsetr = -min_rvalue * scaling_factorr
# scaled_yr1 = (pred[25:45] * scaling_factorr) + offsetr
# scaled_yr2 = (obs[25:45] * scaling_factorr) + offsetr

# scaled_pred = np.hstack([scaled_yz1, scaled_yk1, scaled_yr1])
# scaled_obs = np.hstack([scaled_yz2, scaled_yk2, scaled_yr2])
# np.save("Scaled_obs.npy", scaled_obs)
# scaling_factor = np.hstack([scaling_factorz, scaling_factork, scaling_factorr])
# np.save("Scaling_factor.npy", scaling_factor)
# offset = np.hstack([offsetz, offsetk, offsetr])
# np.save("Offset.npy", offset)
# Manually calculate the weighted MAE
# abs_diff = np.abs(obs - pred) / sigma
# fract = np.load("fractional_sigma.npy")
abs_diff = np.abs(obs - y)
# abs_diff_old = np.abs(obs - y_old)
# print("Abs diff: ", abs_diff)
# abs_diff = ((pred-obs)**2)/sigma**2

bag_i = abs_diff[0:7] / 7
# bag_iold = abs_diff_old[0:7] / 7
driv_k = abs_diff[7:31] / 24
# driv_kold = abs_diff_old[7:25] / 18
driv_r = abs_diff[31:51] / 20
# driv_rold = abs_diff_old[25:45] / 20

mae = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))
# mae = np.mean(abs_diff)
print("\n")
print("Unweighted MAE: ", mae)
print("Redshift distribution MAE: ", np.mean(abs_diff[0:7]))
print("K-band Luminosity function MAE: ", np.mean(abs_diff[7:31]))
print("r-band Luminosity function MAE: ", np.mean(abs_diff[31:51]))

# Convert to Lagrangian likelihood
likelihood = (1/(2*0.05)) * np.exp(-mae / 0.05)
print("Likelihood (b=0.05): ", likelihood)

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
cum_error_z = np.cumsum(bag_i)
# cum_error_z_old = np.cumsum(bag_iold)
cum_error_k = np.cumsum(driv_k)
# cum_error_k_old = np.cumsum(driv_kold)
cum_error_r = np.cumsum(driv_r)
# cum_error_r_old = np.cumsum(driv_rold)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# color = "tab:blue"
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label="Bagley et al. 2020")
l2z = axs[0].plot(bins[0:7], yz1, '--', color='green', label=ratio +' Galform prediction')
# l2zo = axs[0].plot(bins[0:7], y_old[0:7], '--', color='red', label=ratio_old + ' Galfom prediction')
axs[0].tick_params(axis='y')
axs[0].set_xlabel("Redshift, z")
axs[0].set_ylabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$]")
# axs[1].errorbar(df_k["Mag"], df_k["LF"], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='black',
#              ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
axs[1].errorbar(df_ck['Mag'], df_ck['PhiK'], yerr=(df_ck['errorK_lower'], df_ck['errorK_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Cole et al. 2001')
l2k = axs[1].plot(bins[7:31], yk1, '--', color='green', label=ratio + ' Galform prediction')
# l2ko = axs[1].plot(bins[7:25], y_old[7:25], '--', color='red', label=ratio_old + ' Galform prediction')
axs[1].tick_params(axis='y')
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].errorbar(df_r["Mag"], df_r["LF"], yerr=(df_r['error_lower'], df_r['error_upper']), markeredgecolor='black',
             ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
l2r = axs[2].plot(bins[31:51], yr1, '--', color='green', label=ratio + ' Galform prediction')
# l2ro = axs[2].plot(bins[25:45], y_old[25:45], '--', color='red', label=ratio_old + ' Galform prediction')
axs[2].tick_params(axis='y')
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")

axsz = axs[0].twinx()
axsk = axs[1].twinx()
axsr = axs[2].twinx()
# color = "tab:red"
l3z = axsz.step(xz2, cum_error_z, color='green', where='mid', label=ratio + ' cumulative error', alpha=0.5)
# l3zo = axsz.step(xz2, cum_error_z_old, color='red', where='mid', label=ratio_old + ' cumulative error', alpha=0.5)
l3k = axsk.step(xk2, cum_error_k, color='green', where='mid', label=ratio + " cumulative error", alpha=0.5)
# l3ko = axsk.step(xk2, cum_error_k_old, color='red', where='mid', label=ratio_old + " cumulative error", alpha=0.5)
l3r = axsr.step(xr2, cum_error_r, color='green', where='mid', label=ratio + " cumulative error", alpha=0.5)
# l3ro = axsr.step(xr2, cum_error_r_old, color='red', where='mid', label=ratio_old + " cumulative error", alpha=0.5)
axsz.tick_params(axis='y')
axsk.tick_params(axis='y')
axsr.tick_params(axis='y')
axsz.set_ylabel("Cumulative error (MAE)")
axsk.set_ylabel("Cumulative error (MAE)")
axsr.set_ylabel("Cumulative error (MAE)")
legz = l2z + l3z  # + l2zo  + l3zo
labsz = [l.get_label() for l in legz]
axs[0].legend(legz, labsz, loc=6)
legk = l2k + l3k  # + l2ko  # + l3ko
labsk = [l.get_label() for l in legk]
axs[1].legend(legk, labsk, loc=0)
legr = l2r + l3r  # + l2ro  # + l3ro
labsr = [l.get_label() for l in legr]
axs[2].legend(legr, labsr, loc=0)
fig.tight_layout()
plt.show()
