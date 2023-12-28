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
# cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
# cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
# df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00-1.87, mag_high=-16.00-1.87)
# df_ck = df_ck[df_ck['PhiK'] != 0]
# df_ck = df_ck.sort_values(['Mag'], ascending=[True])
# df_ck['errorK_upper'] = np.log10(df_ck['PhiK'] + df_ck['errorK']) - np.log10(df_ck['PhiK'])
# df_ck['errorK_lower'] = np.log10(df_ck['PhiK']) - np.log10(df_ck['PhiK'] - df_ck['errorK'])
# df_ck['PhiK'] = np.log10(df_ck['PhiK'])
# df_ck['Mag'] = df_ck['Mag'] + 1.87

# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
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
X_rand1 = np.array([7.99637597e-01, 2.98758029e+02, 9.25382007e+01, 3.98830512e+00, 8.72572091e-01, 6.94198272e-01,
                    1.14516611e+00, 2.10853713e-01, 2.01385138e-01, 2.96426914e-02, 1.26986318e-01])
ratio1 = "1:1:1 ratio"
X_rand2 = np.array([6.76412161e-01, 3.08220574e+02, 6.41642100e+01, 3.94078882e+00, 6.94554457e-01, 7.02845529e-01,
                    1.18687776e+00, 3.43773888e-01, 1.17092698e-01, 1.23803658e-02, 1.66454001e-01])
ratio2 = "2:1:1 ratio"
X_rand3 = np.array([1.06548467e+00, 2.95916775e+02, 1.45602268e+01, 3.98671662e+00, 8.97507177e-01, 3.79646808e+00,
                    8.68264990e-01, 3.91806079e-01, 2.69824210e-01,  1.30222411e-02, 7.62827444e-02])
ratio3 = "3:1:1 ratio"
X_rand4 = np.array([1.25862247e+00, 2.55638136e+02, 7.80601934e+02, 3.96796948e+00, 6.82960562e-01, 2.78351248e+00,
                    1.01648796e+00, 3.96383673e-01, 2.50095495e-02, 4.31807698e-02, 5.48705862e-02])
ratio4 = "4:1:1 ratio"
X_rand6 = np.array([3.47528918e-01, 1.99275428e+02, 7.03992287e+02, 3.96297958e+00, 7.62162827e-01, 3.84172460e+00,
                    7.89514160e-01, 3.90781641e-01, 4.98376886e-02, 4.90996882e-02, 6.62545384e-02])
ratio6 = "6:1:1 ratio"
# Lacey et al. 2016
# X_rand = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
X_rand1 = X_rand1.reshape(1, -1)

members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
# Load in the array of models and average over the predictions
y1 = np.mean([model(X_rand1) for model in members], axis=0)
y2 = np.mean([model(X_rand2) for model in members], axis=0)
y3 = np.mean([model(X_rand3) for model in members], axis=0)
y4 = np.mean([model(X_rand4) for model in members], axis=0)
y6 = np.mean([model(X_rand6) for model in members], axis=0)

y1 = y1[0]
y2 = y2[0]
y3 = y3[0]
y4 = y4[0]
y6 = y6[0]


# Redshift distribution
# Perform interpolation
yz1 = y1[0:49]
xz2 = Ha_b['z'].values
yz2 = Ha_b['n'].values

# Interpolate or resample the daa onto the common x-axis
# interp_funcz = interp1d(xz1, yz1, kind='linear', fill_value='extrapolate')
# interp_yz1 = interp_funcz(xz2)

# Working out the MAE values
# weighted_maez = mean_absolute_error(yz2, yz1)
# print("MAE redshift distribution: ", weighted_maez)

# Try on the Driver et al. 2012 K-band LF data
# Perform interpolation
yk1 = y1[49:74]
xk2 = df_k['Mag'].values
yk2 = df_k['LF'].values

# Interpolate or resample the data onto the common x-axis
# interp_funck = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
# interp_yk1 = interp_funck(xk2)

# weighted_maek = mean_absolute_error(yk2, yk1)
# print("MAE K-band luminosity function: ", weighted_maek)

# Try on the Driver et al. 2012 r-band LF data
# Perform interpolation
yr1 = y1[74:102]
xr2 = df_r['Mag'].values
yr2 = df_r['LF'].values

# Interpolate or resample the data onto the common x-axis
# interp_funcr = interp1d(xr1, yr1, kind='linear', fill_value='extrapolate')
# interp_yr1 = interp_funcr(xr2)

# weighted_maer = mean_absolute_error(yr2, yr1)
# print("MAE r-band luminosity function: ", weighted_maer)

# Plot to see how this looks
fig, axs = plt.subplots(3, 1, figsize=(8, 14))
axs[0].plot(bins[0:49], yz1, '--', c='tab:blue', label=ratio1)
axs[0].plot(bins[0:49], y2[0:49], '--', c='tab:orange', label=ratio2)
axs[0].plot(bins[0:49], y3[0:49], '--', c='tab:red', label=ratio3)
axs[0].plot(bins[0:49], y4[0:49], '--', c='tab:green', label=ratio4)
axs[0].plot(bins[0:49], y6[0:49], '--', c='tab:purple', label=ratio6)
# axs[0].plot(bins_l[0:49], Lacey_y[0:49], 'r--', label="Lacey et al. 2016", alpha=0.5)
# axs[0].plot(xz2, interp_yz1, 'bx', label='Interpolated galform')
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz [deg$^{-2}$])')
axs[0].set_xlim(0.7, 2.0)
axs[0].legend()
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1].plot(bins[49:74], yk1, '--', c='tab:blue', label='Galform prediction ' + ratio1)
axs[1].plot(bins[49:74], y2[49:74], '--', c='tab:orange', label="Galform prediction " + ratio2)
axs[1].plot(bins[49:74], y3[49:74], '--', c='tab:red', label="Galform prediction " + ratio3)
axs[1].plot(bins[49:74], y4[49:74], '--', c='tab:green', label="Galform prediction " + ratio4)
axs[1].plot(bins[49:74], y6[49:74], '--', c='tab:purple', label="Galform prediction " + ratio6)
# axs[1].plot(bins_l[49:69], Lacey_y[49:69], 'r--', label="Lacey et al. 2016", alpha=0.5)
# axs[1].plot(xk2, interp_yk1, 'bx', label='Interpolated galform')
axs[1].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_xlim(-15, -25)
axs[1].set_ylim(-6.2, -1)
#axs[1].legend()
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].plot(bins[74:], yr1, '--', c='tab:blue', label='Galform prediction ' + ratio1)
axs[2].plot(bins[74:], y2[74:], '--', c='tab:orange', label="Galform prediction " + ratio2)
axs[2].plot(bins[74:], y3[74:], '--', c='tab:red', label="Galform prediction " + ratio3)
axs[2].plot(bins[74:], y4[74:], '--', c='tab:green', label="Galform prediction " + ratio4)
axs[2].plot(bins[74:], y6[74:], '--', c='tab:purple', label="Galform prediction " + ratio6)
# axs[2].plot(bins_l[69:91], Lacey_y[69:91], 'r--', label="Lacey et al. 2016", alpha=0.5)
# axs[2].plot(xr2, interp_yr1, 'bx', label='Interpolated galform')
axs[2].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-14, -25)
axs[2].set_ylim(-6, -1)
# axs[2].legend()
plt.tight_layout()
plt.savefig('Plots/weighting_bestfits.pdf')
plt.show()

exit()
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
abs_diff = np.abs(obs - y1)
# abs_diff_old = np.abs(obs - y_old)
# print("Abs diff: ", abs_diff)
# abs_diff = ((pred-obs)**2)/sigma**2

bag_i = abs_diff[0:7] / 7
# bag_iold = abs_diff_old[0:7] / 7
driv_k = abs_diff[7:25] / 18
# driv_kold = abs_diff_old[7:25] / 18
driv_r = abs_diff[25:45] / 20
# driv_rold = abs_diff_old[25:45] / 20

mae = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))
# mae = np.mean(abs_diff)
print("\n")
print("Unweighted MAE: ", mae)
print("Redshift distribution MAE: ", np.mean(abs_diff[0:7]))
print("K-band Luminosity function MAE: ", np.mean(abs_diff[7:25]))
print("r-band Luminosity function MAE: ", np.mean(abs_diff[25:45]))

# Convert to Lagrangian likelihood
likelihood = (1 / (2 * 0.05)) * np.exp(-mae / 0.05)
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
l2z = axs[0].plot(bins[0:7], yz1, '--', color='green', label=ratio1 + ' Galform prediction')
# l2zo = axs[0].plot(bins[0:7], y_old[0:7], '--', color='red', label=ratio_old + ' Galfom prediction')
axs[0].tick_params(axis='y')
axs[0].set_xlabel("Redshift, z")
axs[0].set_ylabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$]")
# axs[1].errorbar(df_k["Mag"], df_k["LF"], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='black',
#              ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
l2k = axs[1].plot(bins[7:25], yk1, '--', color='green', label=ratio1 + ' Galform prediction')
# l2ko = axs[1].plot(bins[7:25], y_old[7:25], '--', color='red', label=ratio_old + ' Galform prediction')
axs[1].tick_params(axis='y')
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].errorbar(df_r["Mag"], df_r["LF"], yerr=(df_r['error_lower'], df_r['error_upper']), markeredgecolor='black',
                ecolor="black", capsize=2, fmt='co', label="Driver et al. 2012")
l2r = axs[2].plot(bins[25:45], yr1, '--', color='green', label=ratio1 + ' Galform prediction')
# l2ro = axs[2].plot(bins[25:45], y_old[25:45], '--', color='red', label=ratio_old + ' Galform prediction')
axs[2].tick_params(axis='y')
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")

axsz = axs[0].twinx()
axsk = axs[1].twinx()
axsr = axs[2].twinx()
# color = "tab:red"
l3z = axsz.step(xz2, cum_error_z, color='green', where='mid', label=ratio1 + ' cumulative error', alpha=0.5)
# l3zo = axsz.step(xz2, cum_error_z_old, color='red', where='mid', label=ratio_old + ' cumulative error', alpha=0.5)
l3k = axsk.step(xk2, cum_error_k, color='green', where='mid', label=ratio1 + " cumulative error", alpha=0.5)
# l3ko = axsk.step(xk2, cum_error_k_old, color='red', where='mid', label=ratio_old + " cumulative error", alpha=0.5)
l3r = axsr.step(xr2, cum_error_r, color='green', where='mid', label=ratio1 + " cumulative error", alpha=0.5)
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
