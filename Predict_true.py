import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from Loading_functions import dndz_df, lf_df
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

# Import the GALFORM data
dndz_path = 'Data/Data_for_ML/raw_dndz_HaNIIext_bestgalform/dndz_HaNII_ext'
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
df_z = dndz_df(dndz_path, columns_Z)

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
LF_path = 'Data/Data_for_ML/raw_kband_training/LF_bestgalform/gal.lf'
df_lfk = lf_df(LF_path, columns_lf, mag_low=-23.80, mag_high=-15.04)
df_lfk = df_lfk[df_lfk['Krdust'] != 0]
df_lfr = lf_df(LF_path, columns_lf, mag_low=-23.36, mag_high=-13.73)
df_lfr = df_lfr[df_lfr['Rrdust'] != 0]

# Import bin data
bin_file = 'Data/Data_for_ML/bin_data/bin_full_int'
bins = genfromtxt(bin_file)

X_pred = np.array([3.61814487e-01, 2.06151736e+02, 7.24779817e+02, 3.99320682e+00, 6.23875493e-01, 3.67976303e+00,
                   9.79112568e-01, 2.18810180e-01, 2.22115992e-02, 4.32502588e-02, 1.59604685e-01])
X_pred = X_pred.reshape(1, -1)
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))
y = np.mean([model(X_pred) for model in members], axis=0)

y = y[0]

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(bins[0:7], y[0:7], 'b--', label="MCMC prediction")
axs[0].plot(df_z['z'], df_z['dN(>S)/dz'], 'b-', label="True Galform")
axs[0].errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
                fmt='co', label=r"Bagley et al. 2020")
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
axs[0].set_xlim(0.9, 1.6)
axs[0].legend()
axs[1].plot(bins[7:25], y[7:25], 'b--', label="MCMC prediction")
axs[1].plot(df_lfk['Mag'], np.log10(df_lfk['Krdust']), 'b-', label="True Galform")
axs[1].errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_xlim(-16, -24.5)
axs[1].set_ylim(-6.2, -1)
axs[1].legend()
axs[2].plot(bins[25:45], y[25:45], 'b--', label="MCMC prediction")
axs[2].plot(df_lfr['Mag'], np.log10(df_lfr['Rrdust']), 'b-', label="True Galform")
axs[2].errorbar(df_r['Mag'], df_r['LF'], yerr=(df_r['error_lower'], df_r['error_upper']),
                markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlabel(r"M$_{r,AB}$ - 5log(h)")
axs[2].set_xlim(-14, -24)
axs[2].set_ylim(-6, -1)
axs[2].legend()
plt.show()

# PLot on top of true vs predicted plots