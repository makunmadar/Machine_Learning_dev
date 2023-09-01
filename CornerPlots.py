import numpy as np
from numpy import genfromtxt
import corner
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from Loading_functions import lf_df
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

# redshift_samples = np.load("Samples_redshiftdist.npy")
# KLF_samples = np.load("Samples_KLF.npy")
combo_samples = np.load("Samples_combo_MAE811.npy")

# Corner plot
labels = [r"$\alpha_{ret}$", r"$V_{SN, disk}$", r"$V_{SN, burst}$",
          r"$\gamma_{SN}$", r"$\alpha_{cool}$", r"$\nu_{SF}$ [Gyr$^{-1}$]"]
# For multiple walkers the shape of "samples" should have the shape (num_walkers, num_samples, num_parameters)

# flattened_samples = scaler_feat.inverse_transform(flattened_samples)
# Create the corner plot
p_range = [(0, 3.0), (100, 550), (100, 550), (1.5, 3.5), (0.0, 2.0), (0.2, 1.7)]
#
# fig = corner.corner(flattened_zsamples, labels=labels, color='gray',
#                     plot_datapoints=True, levels=[0.68,0.95],
#                     smooth=0.0, bins=50, range=p_range, fill_contours=True)
# corner.corner(flattened_ksamples, labels=labels, color='blue',
#               plot_datapoints=True, levels=[0.68,0.95],
#               smooth=0.0, bins=50, range=p_range, fill_contours=True, fig=fig)
#
# fig.savefig("corner_zLF.png")
#
# figz = corner.corner(flattened_zsamples, show_titles=True, labels=labels, color='gray',
#                     plot_datapoints=True, levels=[0.68,0.95],
#                     smooth=0.0, bins=50, range=p_range, fill_contours=True)
# figz.savefig("corner_z.png")
#
# figk = corner.corner(flattened_ksamples, show_titles=True, labels=labels, color='blue',
#                     plot_datapoints=True, levels=[0.68,0.95],
#                     smooth=0.0, bins=50, range=p_range, fill_contours=True)
# figk.savefig("corner_LF.png")

figc = corner.corner(combo_samples, show_titles=True, labels=labels, color='green',
                     plot_datapoints=True, levels=[0.68, 0.95],
                     smooth=0.0, bins=50, range=p_range, fill_contours=True)
figc.show()
figc.savefig("corner_combo_MAE811.png")

#
# redshift_likelihoods = np.load("Likelihoods_redshiftdist.npy")
# KLF_likelihoods = np.load("Likelihoods_KLF.npy")
combo_likelihoods = np.load("Likelihoods_combo_MAE811.npy")
combo_error = np.load("Error_combo_MAE811.npy")

# Find the best model with the highest likelihood
# max_zlikeli_idx = np.argmax(redshift_likelihoods)
# print("\n")
# print("Highest likelihood dn/dz: ", redshift_likelihoods[max_zlikeli_idx])
# print("Best parameters dn/dz: ", redshift_samples[max_zlikeli_idx])
#
# max_klikeli_idx = np.argmax(KLF_likelihoods)
# print("\n")
# print("Highest likelihood K-LF: ", KLF_likelihoods[max_klikeli_idx])
# print("Best parameters K-LF: ", KLF_samples[max_klikeli_idx])

# max_clikeli_idx = np.argmax(combo_likelihoods)
# print("\n")
# print("Highest likelihood combo: ", combo_likelihoods[max_clikeli_idx])
# print("Corresponding MAE combo: ", combo_error[max_clikeli_idx])
# print("Best parameters combo: ", combo_samples[max_clikeli_idx])
min_cerror_idx = np.argmin(combo_error)
print("\n")
print("Lowest error combo: ", combo_error[min_cerror_idx])
print("Corresponding likelihood combo: ", combo_likelihoods[min_cerror_idx])
print("Best parameters combo: ", combo_samples[min_cerror_idx])

# Load the individual error data
# combo_errorz = np.load("Error_comboz_MAE.npy")
# combo_errork = np.load("Error_combok_MAE.npy")
# bins = np.linspace(0, 5, 50)
#
# plt.hist(combo_errorz, bins=bins, label='Redshift', histtype='step')
# plt.hist(combo_errork, bins=bins, label='LF error', histtype='step')
# plt.xlabel("MAE")
# plt.ylabel("Counts")
# plt.legend()
# plt.ylim([0, 1000])
# plt.show()

# Plotting the predictions
# redshift_predictions = np.load("Predictions_redshiftdist.npy")
# KLF_predictions = np.load("Predictions_KLF.npy")

# Load the Galform bins
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

# Load in the Observational data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]

# Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_pathk = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_pathk, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

drive_pathr = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_pathr, driv_headers)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
df_r['error_upper'] = np.log10(df_r['LF'] + df_r['error']) - np.log10(df_r['LF'])
df_r['error_lower'] = np.log10(df_r['LF']) - np.log10(df_r['LF'] - df_r['error'])
df_r['LF'] = np.log10(df_r['LF'])

# Fitting to redshift distribution
# fig, axs = plt.subplots(1, 2, figsize=(10, 10))
# colour = iter(cm.rainbow(np.linspace(0, 1, len(redshift_predictions))))
#
# for theta in redshift_predictions:
#     c = next(colour)
#     axs[0].plot(bins[0:49], theta[0:49], c=c, alpha=0.1)
#     axs[1].plot(bins[49:67], theta[49:67], c=c, alpha=0.1)
# axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black', capsize=2,
#              fmt='co', label="Bagley et al. 2020")
# axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#              markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')

# axs[0].set_xlabel('Redshift')
# axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
# axs[0].set_xlim(0.7, 2.0)
# axs[0].legend()
# axs[1].set_xlabel(r"M$_{AB}$ - 5log(h)")
# axs[1].set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
# axs[1].set_xlim(-18, -25)
# axs[1].set_ylim(-6, -1)
# plt.show()

# Fitting to Luminosity function distribution
# fig, axs = plt.subplots(1, 2, figsize=(10, 10))
# colour = iter(cm.rainbow(np.linspace(0, 1, len(KLF_predictions))))
#
# for theta in KLF_predictions:
#     c = next(colour)
#     axs[0].plot(bins[0:49], theta[0:49], c=c, alpha=0.1)
#     axs[1].plot(bins[49:67], theta[49:67], c=c, alpha=0.1)
# axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black', capsize=2,
#              fmt='co', label="Bagley et al. 2020")
# axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#              markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')

# axs[0].set_xlabel('Redshift')
# axs[0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
# axs[0].set_xlim(0.7, 2.0)
# axs[0].legend()
# axs[1].set_xlabel(r"M$_{AB}$ - 5log(h)")
# axs[1].set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
# axs[1].set_xlim(-18, -25)
# axs[1].set_ylim(-6, -1)
# plt.show()

combo_predictions_raw = np.load("Predictions_combo_MAE811_raw.npy")

# Combination plots
fig, axs = plt.subplots(5, 3, figsize=(23, 15))
plt.subplots_adjust(hspace=0)

for i in range(len(combo_predictions_raw)):

    colour = iter(cm.rainbow(np.linspace(0, 1, len(combo_predictions_raw[i]))))

    for theta in combo_predictions_raw[i]:
        c = next(colour)
        axs[i, 0].plot(bins[0:49], theta[0:49], c=c, alpha=0.1)
        axs[i, 1].plot(bins[49:67], theta[49:67], c=c, alpha=0.1)
        axs[i, 2].plot(bins[49:67], theta[67:85], c=c, alpha=0.1)
    axs[i, 0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black',
                       capsize=2, fmt='co')
    axs[i, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                       markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
    axs[i, 2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                       markeredgecolor='black', ecolor='black', capsize=2, fmt='co')

    axs[i, 0].set_xlim(0.7, 2.0)
    axs[i, 0].set_ylim(2.5, 4.5)
    axs[i, 1].set_xlim(-18, -25)
    axs[i, 1].set_ylim(-6, -1)
    axs[i, 2].set_xlim(-18, -25)
    axs[i, 2].set_ylim(-6, -1)

axs[0, 0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black',
                   capsize=2, fmt='co', label='Bagley et al. 2020')
axs[0, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                   markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[0, 2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                   markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[0, 0].legend()
axs[0, 1].legend()
axs[0, 2].legend()
axs[2, 0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[2, 1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2, 2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")

axs[-1, 0].set_xlabel('Redshift')
axs[-1, 1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[-1, 2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")

# Add colour bar
iteration_values = np.arange(len(combo_predictions_raw[0]))
scalar_mappable = cm.ScalarMappable(cmap=cm.rainbow)
scalar_mappable.set_array(iteration_values)
colorbar = plt.colorbar(scalar_mappable, ax=axs)
colorbar.set_label('Iteration')

plt.show()
