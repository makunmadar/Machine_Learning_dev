import numpy as np
from numpy import genfromtxt
import corner
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
from Loading_functions import lf_df
import seaborn as sns
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 27
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# ratio = "411_20"
# redshift_samples = np.load("Samples_redshiftdist.npy")
# KLF_samples = np.load("Samples_KLF.npy")
# combo_samples = np.load(f"Samples_combo_MAEup{ratio}.npy")

# Corner plot
# labels = [r"$\alpha_{ret}$", r"$V_{SN, disk}$", r"$V_{SN, burst}$",
#           r"$\gamma_{SN}$", r"$\alpha_{cool}$", r"$\nu_{SF}$",
#           r"$F_{stab}$", r"$f_{ellip}$", r"$f_{burst}$", r"$f_{SMBH}$", r"$\tau_{*burst,min}$"]
# For multiple walkers the shape of "samples" should have the shape (num_walkers, num_samples, num_parameters)

# flattened_samples = scaler_feat.inverse_transform(flattened_samples)
# Create the corner plot
# p_range = [(0.2, 3.0), (10, 800), (10, 800), (1.0, 4.0), (0.0, 4.0), (0.1, 4.0),
#            (0.5, 1.2), (0.2, 0.5), (0.001, 0.3), (0.001, 0.05), (0.01, 0.2)]
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

# figc = corner.corner(combo_samples, show_titles=True, labels=labels, color='green',
#                      plot_datapoints=False, levels=[0.25, 0.50, 0.75],
#                      smooth=1, bins=50, range=p_range, fill_contours=True)
# figc = corner.corner(combo_samples, bins=50, range=p_range, color='green', smooth=3,
#                      labels=labels, show_titles=False, levels=[0.25, 0.50, 0.75],
#                      plot_datapoints=False, fill_contours=True, hist_kwargs={"density": True})
# figc.show()
# figc.savefig(f"corner_combo_MAE{ratio}_smooth2.png")
# exit()
#
# redshift_likelihoods = np.load("Likelihoods_redshiftdist.npy")
# KLF_likelihoods = np.load("Likelihoods_KLF.npy")
# combo_likelihoods = np.load(f"Likelihoods_combo_MAE{ratio}.npy")
# combo_error = np.load(f"Error_combo_MAE{ratio}.npy")

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
# min_cerror_idx = np.argmin(combo_error)
# # print("\n")
# print("Lowest error combo: ", combo_error[min_cerror_idx])
# print("Corresponding likelihood combo: ", combo_likelihoods[min_cerror_idx])
# print("Best parameters combo: ", combo_samples[min_cerror_idx])

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
# bin_file = 'Data/Data_for_ML/bin_data/bin_full_int'
# bins = genfromtxt(bin_file)
#
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
df_k = lf_df(drive_pathk, driv_headers, mag_high=-15.25, mag_low=-23.75)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

drive_pathr = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_pathr, driv_headers, mag_high=-13.75, mag_low=-23.25)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
df_r['error_upper'] = np.log10(df_r['LF'] + df_r['error']) - np.log10(df_r['LF'])
df_r['error_lower'] = np.log10(df_r['LF']) - np.log10(df_r['LF'] - df_r['error'])
df_r['LF'] = np.log10(df_r['LF'])

obs = np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values])

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

# combo_predictions_raw = np.load(f"Predictions_combo_MAE{ratio}_raw.npy")
#
# # Combination plots
# fig, axs = plt.subplots(6, 3, figsize=(23, 15))
# plt.subplots_adjust(hspace=0)
#
# for i in range(len(combo_predictions_raw)):
#
#     colour = iter(cm.rainbow(np.linspace(0, 1, len(combo_predictions_raw[i]))))
#
#     for theta in combo_predictions_raw[i]:
#         c = next(colour)
#         axs[i, 0].plot(bins[0:7], theta[0:7], c=c, alpha=0.1)
#         axs[i, 1].plot(bins[7:25], theta[7:25], c=c, alpha=0.1)
#         axs[i, 2].plot(bins[25:45], theta[25:45], c=c, alpha=0.1)
#     axs[i, 0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black',
#                        capsize=2, fmt='co')
#     # axs[i, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#     #                    markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
#     axs[i, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#                    markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
#     axs[i, 2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
#                        markeredgecolor='black', ecolor='black', capsize=2, fmt='co')
#
#     axs[i, 0].set_xlim(0.9, 1.6)
#     axs[i, 0].set_ylim(2.5, 4.5)
#     axs[i, 1].set_xlim(-15.0, -24)
#     axs[i, 1].set_ylim(-6, -1)
#     axs[i, 2].set_xlim(-13.0, -24)
#     axs[i, 2].set_ylim(-6, -1)
#
# axs[0, 0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black',
#                    capsize=2, fmt='co', label='Bagley et al. 2020')
# axs[0, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#                    markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
# # axs[0, 1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
# #                    markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Cole et al. 2001')
# axs[0, 2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
#                    markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
# axs[0, 0].legend()
# axs[0, 1].legend()
# axs[0, 2].legend()
# axs[2, 0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
# axs[2, 1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
# axs[2, 2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
#
# axs[-1, 0].set_xlabel('Redshift')
# axs[-1, 1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
# axs[-1, 2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
#
# # Add colour bar
# iteration_values = np.arange(len(combo_predictions_raw[0]))
# scalar_mappable = cm.ScalarMappable(cmap=cm.rainbow)
# scalar_mappable.set_array(iteration_values)
# colorbar = plt.colorbar(scalar_mappable, ax=axs)
# colorbar.set_label('Iteration')
#
# plt.show()

# Picking only the minimum MAE from each walker
columns = ["MAE", "alpha_reheat", "vhotdisk", "vhotburst", "alphahot", "alpha_cool",
           "nu_sf", "Fstab", "fellip", "fburst", "fSMBH", "tau_burst"]
minMAE_df = pd.DataFrame(columns=columns)
best_chain_df = pd.DataFrame(columns=columns[1:12])

# walker_no = ['10', '6', '3', '1']
walker_no = ['20', '20v2', '30v3', '30v4']  # , '20v5']
count = 0

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

## 10 Walkers ##
for num in walker_no:

    combo_predictions_raw = np.load(f"Predictions_combo_MAEup411_{num}_raw.npy")
    combo_samples = np.load(f"Samples_combo_MAEup411_{num}.npy")
    combo_WMAE = np.load(f"Error_combo_MAEup411_{num}.npy")
    c = 0  # index for the parameters as they are in one solid array

    for i in range(len(combo_predictions_raw)):
        MAE_chain = []
        samples_chain = []
        for theta in combo_predictions_raw[i]:
            # abs_diff = np.abs(obs - theta)
            # bag = abs_diff[0:7] / 7
            # driv_k = abs_diff[7:25] / 18
            # driv_r = abs_diff[25:45] / 20
            # mae_theta = (np.sum(bag) + np.sum(driv_k) + np.sum(driv_r)) * (1 / 3)
            # sample_theta = combo_samples[c]
            # sample_WMAE = combo_WMAE[c]

            MAE_chain.append(combo_WMAE[c])
            samples_chain.append(combo_samples[c])
            c += 1

        min_index = MAE_chain.index(min(MAE_chain))
        minMAE_chain = MAE_chain[min_index]
        minsamples_chain = samples_chain[min_index]
        minMAE_chain_df = pd.DataFrame([{'MAE': minMAE_chain[0], 'alpha_reheat': minsamples_chain[0],
                                         'vhotdisk': minsamples_chain[1], 'vhotburst': minsamples_chain[2],
                                         'alphahot': minsamples_chain[3], 'alpha_cool': minsamples_chain[4],
                                         'nu_sf': minsamples_chain[5], 'Fstab': minsamples_chain[6],
                                         'fellip': minsamples_chain[7], 'fburst': minsamples_chain[8],
                                         'fSMBH': minsamples_chain[9], 'tau_burst': minsamples_chain[10]}])
        axs[0].plot(bins[0:49], combo_predictions_raw[i][min_index][0:49], alpha=0.1)
        axs[1].plot(bins[49:74], combo_predictions_raw[i][min_index][49:74], alpha=0.1)
        axs[2].plot(bins[74:102], combo_predictions_raw[i][min_index][74:102], alpha=0.1)

        minMAE_df = pd.concat([minMAE_df, minMAE_chain_df], ignore_index=True)

axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black',
                   capsize=2, fmt='co', label='Bagley et al. 2020')
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                   markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                   markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1].set_xlim([-15.0, -26])
axs[2].set_xlim([-15.0, -26])
plt.tight_layout()
plt.show()

#         if count == 2:
#             for j in samples_chain:
#                 best_chain = pd.DataFrame([{'alpha_reheat': j[0],
#                                             'vhotdisk': j[1], 'vhotburst': j[2],
#                                             'alphahot': j[3], 'alpha_cool': j[4],
#                                             'nu_sf': j[5], 'Fstab': j[6],
#                                             'fellip': j[7], 'fburst': j[8],
#                                             'fSMBH': j[9], 'tau_burst': j[10]}])
#                 best_chain_df = pd.concat([best_chain_df, best_chain], ignore_index=True)
#         count += 1
# print(minMAE_df)
# print(best_chain_df)

# iz_list = [271, 194, 182, 169, 152, 142]
# nvol_list = [1, 2, 3, 4, 5]
#
# extra_columns = ['redshift', 'subvolume', 'modelno']
# columns.extend(extra_columns)
# data = []
# model_num = 0
#
# for i in range(len(minMAE_df)):
#     model_num += 1
#     for j in range(len(iz_list)):
#         for k in range(len(nvol_list)):
#
#             row = minMAE_df.iloc[i].tolist()
#             row.append(iz_list[j])
#             row.append(nvol_list[k])
#             row.append(model_num)
#             data.append(row)
#
# minMAE_df_new = pd.DataFrame(columns=columns, data=data)
# # print(minMAE_df_new)
# print(minMAE_df_new['MAE'].min())
# print(minMAE_df_new['MAE'].max())
# minMAE_df_new.to_csv('minMAE_100MCMC_v1.csv', sep=',', index=False)

# Add luminosity limit
# z = 1.00678 -> 176
# z = 2.00203 -> 142

iz_list = [194, 182, 169, 152, 146, 142]  # Euclid
# iz_list = [176, 169, 152, 146, 142]  # WFIRST
# lum_lim = [20.589, 38.762, 70.908, 163.088, 227.179, 280.745]  # Euclid
mag_lim = [-18.34, -19.02, -19.68, -20.58, -20.94, -21.17]
# lum_lim = [25.84, 35.45, 81.54, 113.59, 140.37]  # WFIRST

nvol_list = [1, 2, 3, 4, 5]

extra_columns = ['redshift', 'subvolume', 'modelno', 'mag_lim']
columns.extend(extra_columns)
data = []
model_num = 0

for i in range(len(minMAE_df)):
    model_num += 1
    for j in range(len(iz_list)):
        for k in range(len(nvol_list)):

            row = minMAE_df.iloc[i].tolist()
            row.append(iz_list[j])
            row.append(nvol_list[k])
            row.append(model_num)
            row.append(mag_lim[j])
            data.append(row)

minMAE_df_new = pd.DataFrame(columns=columns, data=data)
minMAE_df_new.to_csv('minMAE_100MCMC_MAGLIM_EUCLID.csv', sep=',', index=False)

exit()
# Corner plot
# idx = minMAE_df['MAE'].idxmin()
minMAE_df = minMAE_df.drop(['MAE'], axis=1)
minMAE_samples = minMAE_df.to_numpy()
#
# labels = [r"$\alpha_{ret}$", r"$V_{SN, disk}$", r"$V_{SN, burst}$",
#           r"$\gamma_{SN}$", r"$\alpha_{cool}$", r"$\nu_{SF}$]",
#           r"$F_{stab}$", r"$f_{ellip}$", r"$f_{burst}$", r"$f_{SMBH}$", r"$\tau_{*burst,min}$"]
# # For multiple walkers the shape of "samples" should have the shape (num_walkers, num_samples, num_parameters)
#
# # Create the corner plot
# scatter_matrix(minMAE_df, alpha=0.8, figsize=(15, 15), diagonal='hist')
sns.set(style="ticks")
# min_df = pd.DataFrame(minMAE_df.loc[idx]).transpose()
pairplot_all = sns.pairplot(minMAE_df, kind="scatter", diag_kind="kde", corner=True, height=1.5)

# scatter_kws_overlay = {'s': 30, 'color': 'red', 'alpha': 1}
# for i, col in enumerate(min_df.columns):
#     for j in range(i + 1, len(min_df.columns)):
#         pairplot_all.axes[j, i].scatter(min_df.iloc[:, i], min_df.iloc[:, j], **scatter_kws_overlay)

# scatter_kws_overlay = {'s': 20, 'color': 'gray', 'alpha': 0.3, 'zorder': 0}
# for i, col in enumerate(best_chain_df.columns):
#     for j in range(i + 1, len(best_chain_df.columns)):
#         pairplot_all.axes[j, i].scatter(best_chain_df.iloc[:, i], best_chain_df.iloc[:, j], **scatter_kws_overlay)
plt.show()
