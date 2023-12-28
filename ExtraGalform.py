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
plt.rcParams["font.size"] = 13
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)


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

# Import Cole et al. 2001
cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
df_ck = lf_df(cole_path_k, cole_headers, mag_low=-25.00-1.87, mag_high=-18.00-1.87)
df_ck = df_ck[df_ck['PhiK'] != 0]
df_ck = df_ck.sort_values(['Mag'], ascending=[True])
df_ck['Mag'] = df_ck['Mag'] + 1.87
df_ck['error_upper'] = np.log10(df_ck['PhiK'] + df_ck['errorK']) - np.log10(df_ck['PhiK'])
df_ck['error_lower'] = np.log10(df_ck['PhiK']) - np.log10(df_ck['PhiK'] - df_ck['errorK'])
df_ck['PhiK'] = np.log10(df_ck['PhiK'])




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

Lacey_y = np.load("Lacey_y_true.npy")
bins_l = np.load('Lacey_bins.npy')


K_i = [j for j, e in enumerate(list_LFK[61]) if e != 0]
plt.plot(LFK_bins[K_i], list_LFK[61][K_i], '-', c='tab:red', label='Best-fitting GALFORM')
plt.errorbar(df_ck['Mag'], df_ck['PhiK'], yerr=(df_ck['error_lower'], df_ck['error_upper']), markeredgecolor='tab:blue',
                ecolor='tab:blue', capsize=2, fmt='o', label='Cole et al. 2001', zorder=11, markerfacecolor="None")
plt.errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']), markeredgecolor='tab:orange',
                ecolor='tab:orange', capsize=2, fmt='o', label='Driver et al. 2012', zorder=11, markerfacecolor="None")
plt.plot(bins_l[49:69], Lacey_y[49:69], '-', c='tab:gray', label="Lacey et al. 2016", alpha=0.7)
plt.ylabel(r"log$_{10}$($\phi$ (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
plt.xlabel(r"M$_{K,AB}$ - 5log(h)")
plt.xlim(-16, -24.6)
plt.ylim(-6, -1)
plt.legend()
plt.tight_layout()
plt.savefig("Plots/BestvsCole01.pdf")
plt.show()

# Calculate the MAE compared to the Cole01 model
interp_funck = interp1d(LFK_bins[K_i], list_LFK[61][K_i], kind='linear', fill_value='extrapolate')
interp_yk = interp_funck(df_ck['Mag'])

interp_funclk = interp1d(bins_l[49:69], Lacey_y[49:69], kind='linear', fill_value='extrapolate')
interp_ylk = interp_funclk(df_ck['Mag'])

k_diff = np.abs(df_ck['PhiK'].values - interp_yk)
k_mae = (1/26) * np.sum(k_diff)
print('\nK-band LF MAE: ', k_mae)

lk_diff = np.abs(df_ck['PhiK'].values - interp_ylk)
lk_mae = (1/26) * np.sum(lk_diff)
print('\nLacey16 K-band LF MAE: ', lk_mae)



