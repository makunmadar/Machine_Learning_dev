from Loading_functions import predict_all_models
import os
import re
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from Loading_functions import dndz_df, lf_df, load_all_models, dndz_generation, LF_generation
from scipy.interpolate import interp1d
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
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

# Import the GALFORM data
MCMCdndz_path = 'Data/Data_for_ML/MCMC/dndz_HaNII_ext_20/'
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
MCMCdndz_filenames = os.listdir(MCMCdndz_path)
MCMCdndz_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
list_dndz = np.empty((0, 49))
for file in MCMCdndz_filenames:
    df_z = dndz_df(MCMCdndz_path + file, columns_Z)
    list_dndz = np.vstack(([list_dndz, df_z['dN(>S)/dz'].values]))
dndz_bins = df_z['z'].values


columns_lf = ['Mag', 'Ur', 'Ur(error)', 'Urdust', 'Urdust(error)',
              'Br', 'Br(error)', 'Brdust', 'Brdust(error)',
              'Vr', 'Vr(error)', 'Vrdust', 'Vrdust(error)',
              'Rr', 'Rr(error)', 'Rrdust', 'Rrdust(error)',
              'Ir', 'Ir(error)', 'Irdust', 'Irdust(error)',
              'Jr', 'Jr(error)', 'Jrdust', 'Jrdust(error)',
              'Hr', 'Hr(error)', 'Hrdust', 'Hrdust(error)',
              'Kr', 'Kr(error)', 'Krdust', 'Krdust(error)',
              'Bjr', 'Bjr(error)', 'Bjrdust', 'Bjrdust(error)',
              'Uo', 'Uo(error)', 'Uodust', 'Uodust(error)',
              'Bo', 'Bo(error)', 'Bodust', 'Bodust(error)',
              'Vo', 'Vo(error)', 'Vodust', 'Vodust(error)',
              'Ro', 'Ro(error)', 'Rodust', 'Rodust(error)',
              'Io', 'Io(error)', 'Iodust', 'Iodust(error)',
              'Jo', 'Jo(error)', 'Jodust', 'Jodust(error)',
              'Ho', 'Ho(error)', 'Hodust', 'Hodust(error)',
              'Ko', 'Ko(error)', 'Kodust', 'Kodust(error)',
              'Bjo', 'Bjo(error)', 'Bjodust', 'Bjodust(error)',
              'LCr', 'LCr(error)', 'LCrdust', 'LCrdust(error)'
              ]
MCMCLF_path = 'Data/Data_for_ML/MCMC/LF_20/'
MCMCLF_filenames = os.listdir(MCMCLF_path)
MCMCLF_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
list_LFK = np.empty((0, 21))
list_LFr = np.empty((0, 23))
for file in MCMCLF_filenames:
    df_lfk = lf_df(MCMCLF_path + file, columns_lf, mag_low=-23.80, mag_high=-15.04)
    df_lfk['Krdust'] = np.log10(df_lfk['Krdust'].mask(df_lfk['Krdust'] <= 0)).fillna(0)
    df_lfr = lf_df(MCMCLF_path + file, columns_lf, mag_low=-23.36, mag_high=-13.73)
    df_lfr['Rrdust'] = np.log10(df_lfr['Rrdust'].mask(df_lfr['Rrdust'] <= 0)).fillna(0)

    list_LFK = np.vstack([list_LFK, df_lfk['Krdust'].values])
    list_LFr = np.vstack([list_LFr, df_lfr['Rrdust'].values])
LFK_bins = df_lfk['Mag'].values
LFR_bins = df_lfr['Mag'].values

# Find the MAE and work out the index of lowest MAE
Bgdndz, _ = dndz_generation(galform_filenames=MCMCdndz_filenames, galform_filepath=MCMCdndz_path,
                            O_df=Ha_b, column_headers=columns_Z)
Drlf = LF_generation(galform_filenames=MCMCLF_filenames, galform_filepath=MCMCLF_path,
                     O_dfk=df_k, O_dfr=df_r, column_headers=columns_lf)

obs_y = np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values])
gal_y = np.hstack([Bgdndz, Drlf])
list_mae = []
for i in range(len(gal_y)):
    abs_diff = np.abs(obs_y - gal_y[i])
    bag_i = abs_diff[0:7] / 7
    driv_k = abs_diff[7:25] / 18
    driv_r = abs_diff[25:45] / 20
    mae = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))
    list_mae.append(mae)
minMAE_idx = list_mae.index(min(list_mae))
print('Model with min MAE: ', minMAE_idx+1)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for i in range(20):
    axs[0].plot(dndz_bins, list_dndz[i], 'b-', alpha=0.3)

    K_i = [j for j, e in enumerate(list_LFK[i]) if e != 0]
    axs[1].plot(LFK_bins[K_i], list_LFK[i][K_i], 'b-', alpha=0.3)

    r_i = [j for j, e in enumerate(list_LFr[i]) if e != 0]
    axs[2].plot(LFR_bins[r_i], list_LFr[i][r_i], 'b-', alpha=0.3)

    if i == minMAE_idx:
        axs[0].plot(dndz_bins, list_dndz[i], 'r-', zorder=10, lw=3, label='Best Galform')
        axs[1].plot(LFK_bins[K_i], list_LFK[i][K_i], 'r-', zorder=10, lw=3, label='Best Galform')
        axs[2].plot(LFR_bins[r_i], list_LFr[i][r_i], 'r-', zorder=10, lw=3, label='Best Galform')

axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020", zorder=11)
axs[0].set_xlim(0.8, 1.7)
axs[0].set_ylim(0, 5)
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[0].legend()
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='black',
                ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012', zorder=11)
axs[1].set_xlim(-15, -24.5)
axs[1].set_ylim(-6, -1)
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].legend()
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']), markeredgecolor='black',
                ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012', zorder=11)
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].legend()
plt.show()

# Want to plot the parameters used and their spread

# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full_int'
bins = genfromtxt(bin_file)

X_pred = np.array([0.227634788603344, 203.276437839648, 798.291115969626, 3.98816130426919, 0.688715863150072,
                   3.58219345626837, 1.02557484534723, 0.280960576119976, 0.110700159331381, 0.010497577648371,
                   0.14873019910089])

X_pred = X_pred.reshape(1, -1)
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
y = np.mean([model(X_pred) for model in members], axis=0)

y = y[0]

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(bins[0:7], y[0:7], 'r--', label="Emulator prediction")
axs[0].plot(dndz_bins, list_dndz[9], 'r-', label='True Galform')
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[0].set_xlim(0.8, 1.7)
axs[0].set_ylim(2.2, 4.1)
axs[0].legend()
axs[1].plot(bins[7:25], y[7:25], 'r--', label="Emulator prediction")
K_i = [j for j, e in enumerate(list_LFK[9]) if e != 0]
axs[1].plot(LFK_bins[K_i], list_LFK[9][K_i], 'r-', label='True Galform')
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_xlim(-15, -24.5)
axs[1].set_ylim(-6, -1)
axs[1].legend()
axs[2].plot(bins[25:45], y[25:45], 'r--', label="Emulator prediction")
r_i = [j for j, e in enumerate(list_LFr[9]) if e != 0]
axs[2].plot(LFR_bins[r_i], list_LFr[9][r_i], 'r-', label='True Galform')
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].legend()
plt.show()

# Plot with the True vs Pred plots:
# X_test = np.load('Data/Data_for_ML/testing_data/X_test_100_fullup_int.npy')
# y_test = np.load('Data/Data_for_ML/testing_data/y_test_100_fullup_int.npy')
#
# yhat_all = predict_all_models(n_models=5, X_test=X_test, variant='_9x5_mask_2899_LRELU_int')
# yhat_avg = np.mean(yhat_all, axis=0)
#
# y_testz = [i[0:7] for i in y_test]
# yhatz = [i[0:7] for i in yhat_avg]
# y_testk = [i[7:25] for i in y_test]
# yhatk = [i[7:25] for i in yhat_avg]
# yhatk = [row_a[row_b != 0] for row_a, row_b in zip(yhatk, y_testk)]
# y_testk = [row[row != 0] for row in y_testk]
# y_testr = [i[25:45] for i in y_test]
# yhatr = [i[25:45] for i in yhat_avg]
# yhatr = [row_a[row_b != 0] for row_a, row_b in zip(yhatr, y_testr)]
# y_testr = [row[row != 0] for row in y_testr]
# yhatz = np.ravel(yhatz)
# y_testz = np.ravel(y_testz)
# yhatk = np.concatenate(yhatk).ravel().tolist()
# y_testk = np.concatenate(y_testk).ravel().tolist()
# yhatr = np.concatenate(yhatr).ravel().tolist()
# y_testr = np.concatenate(y_testr).ravel().tolist()
#
# interp_funcz = interp1d(df_z['z'].values, df_z['dN(>S)/dz'].values, kind='linear', fill_value='extrapolate')
# interp_yz1 = interp_funcz(Ha_b['z'].values)
# interp_yz1[Ha_b['z'].values > max(df_z['z'].values)] = 0
# interp_yz1[Ha_b['z'].values < min(df_z['z'].values)] = 0
#
# df_lfk['Krdust'] = np.log10(df_lfk['Krdust'].mask(df_lfk['Krdust'] <= 0)).fillna(0)
# df_lfk = df_lfk[df_lfk['Krdust'] != 0]
# interp_funck = interp1d(df_lfk['Mag'].values, df_lfk['Krdust'].values, kind='linear', fill_value='extrapolate',
#                         bounds_error=False)
# interp_yk1 = interp_funck(df_k['Mag'].values)
# interp_yk1[df_k['Mag'].values < min(df_lfk['Mag'].values)] = 0
#
# df_lfr['Rrdust'] = np.log10(df_lfr['Rrdust'].mask(df_lfr['Rrdust'] <= 0)).fillna(0)
# df_lfr = df_lfr[df_lfr['Rrdust'] != 0]
# interp_funcr = interp1d(df_lfr['Mag'].values, df_lfr['Rrdust'].values, kind='linear', fill_value='extrapolate',
#                         bounds_error=False)
# interp_yr1 = interp_funcr(df_r['Mag'].values)
# interp_yr1[df_r['Mag'].values < min(df_lfk['Mag'].values)] = 0
#
# lacey_true = np.load('Data/Data_for_ML/Observational/Lacey_16/Lacey_y_true_int.npy')
# lacey_pred = np.load('Data/Data_for_ML/Observational/Lacey_16/Lacey_y_pred.npy')

# fig, axs = plt.subplots(1, 3, figsize=(20, 6))
# axs[0].plot(y_testz, yhatz, '.', markersize=2, alpha=0.5)
# axs[0].plot(interp_yz1, y[0:7], marker='o', color='green', markeredgecolor="black", markersize=4)
# axs[0].plot(lacey_true[0:7], lacey_pred[0:7], marker='o', color='red', markeredgecolor="black", markersize=4)
# axs[0].axline((2, 2), slope=1, color='black', linestyle='dotted')
# axs[0].set_xlabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$] True")
# axs[0].set_ylabel("log$_{10}$(dN(>S)/dz) [deg$^{-2}$] Predict")
# axs[0].set_aspect('equal', 'box')
# axs[0].set_xlim([1.6, 4.6])
# axs[0].set_ylim([1.6, 4.6])
# axs[1].plot(y_testk, yhatk, '.', markersize=2, alpha=0.5)
# axs[1].plot(interp_yk1, y[7:25], marker='o', color='green', markeredgecolor="black", markersize=4)
# axs[1].plot(lacey_true[7:25], lacey_pred[7:25], marker='o', color='red', markeredgecolor="black", markersize=4)
# axs[1].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
# axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h) True")
# axs[1].set_ylabel(r"M$_{K,AB}$ - 5log(h) Predict")
# axs[1].set_aspect('equal', 'box')
# axs[1].set_xlim([-5.9, -0.3])
# axs[1].set_ylim([-5.9, -0.3])
# axs[2].plot(y_testr, yhatr, '.', markersize=2, label='Emulator on test set', alpha=0.5)
# axs[2].plot(interp_yr1, y[25:45], marker='o', color='green', markeredgecolor="black", markersize=4, label='Emulator "best fit"')
# axs[2].plot(lacey_true[25:45], lacey_pred[25:45], marker='o', color='red', markeredgecolor="black", markersize=4, label='Emulator Lacey')
# axs[2].axline((-0.5, -0.5), slope=1, color='black', linestyle='dotted')
# axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h) True")
# axs[2].set_ylabel(r"M$_{r,AB}$ - 5log(h) Predict")
# axs[2].set_aspect('equal', 'box')
# axs[2].set_xlim([-5.9, -0.3])
# axs[2].set_ylim([-5.9, -0.3])
# plt.legend()
# plt.show()


# Working out the MAE:
# obs = np.hstack([interp_yz1, interp_yk1, interp_yr1])
# abs_diff = np.abs(obs - y)
# bag_i = abs_diff[0:7] / 7
# driv_k = abs_diff[7:25] / 18
# driv_r = abs_diff[25:45] / 20
#
# mae = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))
# print("\n")
# print("Unweighted MAE: ", mae)
# print("Redshift distribution MAE: ", np.mean(abs_diff[0:7]))
# print("K-band Luminosity function MAE: ", np.mean(abs_diff[7:25]))
# print("r-band Luminosity function MAE: ", np.mean(abs_diff[25:45]))
