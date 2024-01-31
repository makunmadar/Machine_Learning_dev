from Loading_functions import predict_all_models
import os
import re
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from Loading_functions import dndz_df, lf_df, load_all_models, dndz_generation_int, LF_generation_int
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
MCMCdndz_path = 'Data/Data_for_ML/MCMC/dndz_HaNII_ext_100/'
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
MCMCLF_path = 'Data/Data_for_ML/MCMC/LF_100/'
MCMCLF_filenames = os.listdir(MCMCLF_path)
MCMCLF_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
list_LFK = np.empty((0, 25))
list_LFr = np.empty((0, 28))
for file in MCMCLF_filenames:
    df_lfk = lf_df(MCMCLF_path + file, columns_lf, mag_low=-25.55, mag_high=-15.04)
    df_lfk['Krdust'] = np.log10(df_lfk['Krdust'].mask(df_lfk['Krdust'] <= 0)).fillna(0)
    df_lfr = lf_df(MCMCLF_path + file, columns_lf, mag_low=-25.55, mag_high=-13.73)
    df_lfr['Rrdust'] = np.log10(df_lfr['Rrdust'].mask(df_lfr['Rrdust'] <= 0)).fillna(0)

    list_LFK = np.vstack([list_LFK, df_lfk['Krdust'].values])
    list_LFr = np.vstack([list_LFr, df_lfr['Rrdust'].values])
LFK_bins = df_lfk['Mag'].values
LFR_bins = df_lfr['Mag'].values

# Find the MAE and work out the index of lowest MAE
Bgdndz, _ = dndz_generation_int(galform_filenames=MCMCdndz_filenames, galform_filepath=MCMCdndz_path,
                            O_df=Ha_b, column_headers=columns_Z)
Drlf = LF_generation_int(galform_filenames=MCMCLF_filenames, galform_filepath=MCMCLF_path,
                     O_dfk=df_k, O_dfr=df_r, column_headers=columns_lf)

obs_y = np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values])
gal_y = np.hstack([Bgdndz, Drlf])
list_mae = []
W = [4.0] * 7 + [1.0] * 18 + [1.0] * 20
for i in range(len(gal_y)):
    abs_diff = np.abs(obs_y - gal_y[i])
    weighted_diff = W * abs_diff
    bag_i = weighted_diff[0:7] / 7
    driv_k = weighted_diff[7:25] / 18
    driv_r = weighted_diff[25:45] / 20
    mae = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))
    list_mae.append(mae)

# print("Range of 100: ", min(list_mae), max(list_mae))
arr_mae = np.array(list_mae)
min50 = arr_mae.argsort()[:50]
# np.save("Data/Data_for_ML/MCMC/min50idx.npy", min50)
# print("Range of Best 50:", min(list_mae), list_mae[41])
# print("Range of rest 50:", list_mae[42], max(list_mae))

minMAE_idx = list_mae.index(min(list_mae))
print('Model with min MAE: ', minMAE_idx+1)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for i in min50:
    axs[0].plot(dndz_bins, list_dndz[i], '-', c='tab:blue', alpha=0.3)

    K_i = [j for j, e in enumerate(list_LFK[i]) if e != 0]
    axs[1].plot(LFK_bins[K_i], list_LFK[i][K_i], '-', c='tab:blue', alpha=0.3)

    r_i = [j for j, e in enumerate(list_LFr[i]) if e != 0]
    axs[2].plot(LFR_bins[r_i], list_LFr[i][r_i], '-', c='tab:blue', alpha=0.3)

    if i == minMAE_idx:
        axs[0].plot(dndz_bins, list_dndz[i], '-', c='tab:red', zorder=10, lw=3, label='Best GALFORM')
        axs[1].plot(LFK_bins[K_i], list_LFK[i][K_i], '-', c='tab:red', zorder=10, lw=3, label='Best GALFORM')
        axs[2].plot(LFR_bins[r_i], list_LFr[i][r_i], '-', c='tab:red', zorder=10, lw=3, label='Best GALFORM')

axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020", zorder=11)
axs[0].set_xlim(0.8, 1.7)
axs[0].set_ylim(2.2, 4.5)
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz [deg$^{-2}$])')
axs[0].legend()
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='black',
                ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012', zorder=11)
axs[1].set_xlim(-15, -25)
axs[1].set_ylim(-6, -1)
axs[1].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].legend()
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']), markeredgecolor='black',
                ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012', zorder=11)
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].legend()
plt.tight_layout()
plt.savefig("Plots/MCMCvsGALFORM50.pdf")
plt.show()

# Want to plot the parameters used and their spread
# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

Lacey_y = np.load("Lacey_y_true.npy")
bins_l = np.load('Lacey_bins.npy')

# Calculate MAE of Lacey
interp_funcz = interp1d(bins_l[0:49], Lacey_y[0:49], kind='linear', fill_value='extrapolate')
interp_yz = interp_funcz(Ha_b['z'])
interp_funck = interp1d(bins_l[49:69], Lacey_y[49:69], kind='linear', fill_value='extrapolate')
interp_yk = interp_funck(df_k['Mag'])
interp_funcr = interp1d(bins_l[69:91], Lacey_y[69:91], kind='linear', fill_value='extrapolate')
interp_yr = interp_funcr(df_r['Mag'])
# l16 = np.hstack([interp_yz, interp_yk, interp_yr])
# abs_diffl = np.abs(obs_y - l16)
# weighted_diffl = W * abs_diffl
# bag_i = weighted_diffl[0:7] / 7
# driv_k = weighted_diffl[7:25] / 18
# driv_r = weighted_diffl[25:45] / 20
# mael = (1 / 3) * (np.sum(bag_i) + np.sum(driv_k) + np.sum(driv_r))

X_pred = np.array([0.27309144967986, 201.300933219704, 785.637896370095, 3.98424484172544, 0.790965964180526,
                   3.96637640031259, 0.847244990735446,	0.217501594564047, 0.083111404590309, 0.039434811399415,
                   0.031851833487735])

X_pred = X_pred.reshape(1, -1)
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
y = np.mean([model(X_pred) for model in members], axis=0)

y = y[0]

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(bins[0:49], y[0:49], '--', c='tab:red', label="Emulator prediction")
axs[0].plot(dndz_bins, list_dndz[61], '-', c='tab:red', label='True GALFORM')
axs[0].plot(bins_l[0:49], Lacey_y[0:49], '-', c='tab:gray',label="Lacey et al. 2016", alpha=0.7)
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz [deg$^{-2}$])')
axs[0].set_xlim(0.8, 1.7)
axs[0].set_ylim(2.2, 4.5)
axs[0].legend()
axs[1].plot(bins[49:74], y[49:74], '--', c='tab:red', label="Emulator prediction")
K_i = [j for j, e in enumerate(list_LFK[61]) if e != 0]
axs[1].plot(LFK_bins[K_i], list_LFK[61][K_i], '-', c='tab:red', label='True GALFORM')
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1].plot(bins_l[49:69], Lacey_y[49:69], '-', c='tab:gray', label="Lacey et al. 2016", alpha=0.7)
axs[1].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_xlim(-15, -25)
axs[1].set_ylim(-6, -1)
axs[1].legend()
axs[2].plot(bins[74:], y[74:], '--', c='tab:red', label="Emulator prediction")
r_i = [j for j, e in enumerate(list_LFr[61]) if e != 0]
axs[2].plot(LFR_bins[r_i], list_LFr[61][r_i], '-', c='tab:red', label='True GALFORM')
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].plot(bins_l[69:91], Lacey_y[69:91], '-', c='tab:gray', label="Lacey et al. 2016", alpha=0.7)
axs[2].set_ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].legend()
plt.tight_layout()
plt.savefig("Plots/EmulatorvsGALFORMbestMCMC.pdf")
plt.show()

df = pd.read_csv('minMAE_100MCMC_v1.csv')
df = df.iloc[::30]
print(df.iloc[61])
print("\n")
df = df.iloc[min50]

min_values = df.min()
max_values = df.max()

print("Minimum values for each column:")
print(min_values)
print("\nMaximum values for each column:")
print(max_values)
print("\n")

# Calculate the individual MAE from the best fitting model and the Lacey16 model
gal_yz = gal_y[61][0:7]
gal_yk = gal_y[61][7:25]
gal_yr = gal_y[61][25:]
z_diff = np.abs(Ha_b['n'].values - gal_yz)
z_mae = (1/7) * np.sum(z_diff)
print('Redshift distribution MAE: ', z_mae)
k_diff = np.abs(df_k['LF'].values[0:9] - gal_yk[0:9])  # Modified to only calculate the MAE for the bright end
k_mae = (1/9) * np.sum(k_diff)
print('\nK-band LF MAE: ', k_mae)
r_diff = np.abs(df_r['LF'].values[0:10] - gal_yr[0:10])  # Modified to only calculate the MAE for the bright end
r_mae = (1/10) * np.sum(r_diff)
print('\nr-band LF MAE: ', r_mae)

zl_diff = np.abs(Ha_b['n'].values - interp_yz)
zl_mae = (1/7) * np.sum(zl_diff)
print('\nLacey16 Redshift distribution MAE: ', zl_mae)
kl_diff = np.abs(df_k['LF'].values[0:9] - interp_yk[0:9])  # Modified to only calculate the MAE for the bright end
kl_mae = (1/9) * np.sum(kl_diff)
print('\nLacey16 K-band LF MAE: ', kl_mae)
rl_diff = np.abs(df_r['LF'].values[0:10] - interp_yr[0:10])  # Modified to only calculate the MAE for the bright end
rl_mae = (1/10) * np.sum(rl_diff)
print('\nLacey16 r-band LF MAE: ', rl_mae)
