# Predicting the plots from Lacey et al. 16
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from numpy import genfromtxt
from joblib import load
from sklearn.metrics import mean_absolute_error
from Loading_functions import predict_all_models


def emline_df(path, columns):
    data = []
    with open(path, 'r') as fh:
        for curline in fh:
            if curline.startswith("#"):
                header = curline
            else:
                row = curline.strip().split()
                data.append(row)

    data = np.vstack(data)
    df = pd.DataFrame(data=data)
    df = df.apply(pd.to_numeric)
    df.columns = columns

    df = df[(df['Mag'] <= -17.67)]
    df = df[(df['Mag'] >= -25.11)]
    df.reset_index(drop=True, inplace=True)
    # df['Mag'] = np.log10(df['Mag'].replace(0, np.nan))
    # df['Krdust'] = np.log10(df['Krdust'].replace(0, 1e-20))
    df['Krdust'] = np.log10(df['Krdust'].mask(df['Krdust'] <=0)).fillna(0)

    return df


def dz_df(path):
    flist = open(path).readlines()
    data = []
    parsing = False
    for line in flist:
        if line.startswith("# S_nu/Jy=   2.1049E+07"):
            parsing = True
        elif line.startswith("# S_nu/Jy=   2.3645E+07"):
            parsing = False
        if parsing:
            if line.startswith("#"):
                header = line
            else:
                row = line.strip().split()
                data.append(row)

    data = np.vstack(data)

    with open(path, 'r') as fh:
        for curline in fh:
            if curline.startswith("#"):
                header = curline
            else:
                break

    header = header[1:].strip().split()

    df = pd.DataFrame(data=data, columns=header)
    df = df.apply(pd.to_numeric)
    df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].replace(0, np.nan))
    df = df[df['z'] < 2.1]
    df = df[df['z'] > 0.7]

    return df


#  Lacey parameters
X_test = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
# X_test = np.array([0.69578677, 330.96668774, 293.5104189, 3.058798, 0.68834065, 0.71530994])
X_test = X_test.reshape(1, -1)

# Load scalar fits
# scaler_feat = load("mm_scaler_feat_900.bin")
# X_test = scaler_feat.transform(X_test)
# # Use standard scalar for the label data
# scaler_label = load("std_scaler_label.bin")

# Make predictions on the galform set
yhat_all = predict_all_models(n_models=5, X_test=X_test)
yhat_avg = np.mean(yhat_all, axis=0)

yhatz = yhat_avg[0][0:49]
yhatk = yhat_avg[0][49:67]

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

path_zlc = "Data/Data_for_ML/Observational/Lacey_16/dndz_Bagley_HaNII_ext"
dflc = dz_df(path_zlc)

z_test_lc = dflc['dN(>S)/dz'].values
# z_test_lc = z_test_lc[0::4]

# Manual MAE score
maelc_z = mean_absolute_error(z_test_lc, yhatz)
#
fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(bins[0:49], yhatz, 'b--', label=f"Prediction MAE: {maelc_z:.3f}")
# axs.plot(bins[0:49], yhat_all[0][0,0:49], '--', alpha=0.5, label='Model 1')
# axs.plot(bins[0:49], yhat_all[1][0,0:49], '--', alpha=0.5, label='Model 2')
# axs.plot(bins[0:49], yhat_all[2][0,0:49], '--', alpha=0.5, label='Model 3')
# axs.plot(bins[0:49], yhat_all[3][0,0:49], '--', alpha=0.5, label='Model 4')
# axs.plot(bins[0:49], yhat_all[4][0,0:49], '--', alpha=0.5, label='Model 5')

# Original galform data
dflc.plot(ax=axs, x="z", y="dN(>S)/dz", color='blue', label="Lacey et al. 2016")
# axs.scatter(bins[0:49], z_test_lc, color='blue', marker='x', label="Evaluation bins")
axs.set_ylabel(r"Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
axs.set_xlabel(r"Redshift, z", fontsize=15)
axs.set_xlim(0.7, 2.0)
axs.set_ylim(3.0, 4.0)
plt.tick_params(labelsize=15)
plt.legend()
plt.show()

columns_t = ['Mag', 'Ur', 'Ur(error)', 'Urdust', 'Urdust(error)',
             'Br', 'Br(error)', 'Brdust', 'Brdust(error)',
             'Vr', 'Vr(error)', 'Vrdust', 'Vrdust(error)',
             'Rr', 'Rr(error)', 'Rrdust', 'Rrdust(error)',
             'Ir', 'Ir(error)', 'Irdust', 'Irdust(error)',
             'Jr', 'Jr(error)', 'Jrdust', 'Jrdust(error)',
             'Hr', 'Hr(error)', 'Hrdust', 'Hrdust(error)',
             'Kr', 'Kr(error)', 'Krdust', 'Krdust(error)',
             'Uo', 'Uo(error)', 'Uodust', 'Uodust(error)',
             'Bo', 'Bo(error)', 'Bodust', 'Bodust(error)',
             'Vo', 'Vo(error)', 'Vodust', 'Vodust(error)',
             'Ro', 'Ro(error)', 'Rodust', 'Rodust(error)',
             'Io', 'Io(error)', 'Iodust', 'Iodust(error)',
             'Jo', 'Jo(error)', 'Jodust', 'Jodust(error)',
             'Ho', 'Ho(error)', 'Hodust', 'Hodust(error)',
             'Ko', 'Ko(error)', 'Kodust', 'Kodust(error)',
             'LCr', 'LCr(error)', 'LCrdust', 'LCrdust(error)'
             ]

path_lflc = "Data/Data_for_ML/Observational/Lacey_16/gal.lf"

df_lflc = emline_df(path_lflc, columns_t)

k_test_lc_full = df_lflc['Krdust'].values
k_test_lc_sub = k_test_lc_full
# k_test_lc_sub = k_test_lc_full[0::2]

# Ignore the zero truth values
yhatk_lc_sub = yhatk[k_test_lc_sub != 0]
binsk_lc_sub = bins[49:67][k_test_lc_sub != 0]
k_test_lc_sub = k_test_lc_sub[k_test_lc_sub != 0]

binsk_full = df_lflc['Mag'][k_test_lc_full != 0]
k_test_lc_full = k_test_lc_full[k_test_lc_full != 0]

# Manual MAE score
maelc_k = mean_absolute_error(k_test_lc_sub, yhatk_lc_sub)

fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(bins[49:67], yhatk, 'b--', label=f"Prediction MAE: {maelc_k:.3f}")
# axs.plot(bins[49:67], yhat_all[0][0,49:67], '--', alpha=0.5, label='Model 1')
# axs.plot(bins[49:67], yhat_all[1][0,49:67], '--', alpha=0.5, label='Model 2')
# axs.plot(bins[49:67], yhat_all[2][0,49:67], '--', alpha=0.5, label='Model 3')
# axs.plot(bins[49:67], yhat_all[3][0,49:67], '--', alpha=0.5, label='Model 4')
# axs.plot(bins[49:67], yhat_all[4][0,49:67], '--', alpha=0.5, label='Model 5')

axs.plot(binsk_full, k_test_lc_full, 'b-', label="Lacey et al. 2016")
# axs.scatter(binsk_lc_sub, k_test_lc_sub, color='blue', marker='x', label="Evaluation bins")
axs.set_xlabel(r"M$_{AB}$ - 5log(h)", fontsize=15)
axs.set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)", fontsize=15)
axs.set_xlim(-18, -25)
axs.set_ylim(-6, -1)
plt.tick_params(labelsize=15)
plt.legend()
plt.show()

# MCMC_params = np.array([2.33, 545.51, 227.26, 2.93, 0.69, 0.59])
# MCMC_params_up = np.array([2.48, 551.34, 234.44, 2.95, 0.71, 0.60])
# MCMC_params_low = np.array([2.23, 540.06, 220.73, 2.91, 0.68, 0.58])
#
# MCMC_params = MCMC_params.reshape(1, -1)
# MCMC_params_up = MCMC_params_up.reshape(1, -1)
# MCMC_params_low = MCMC_params_low.reshape(1, -1)
# # Load scalar fits
# scaler_feat = load("mm_scaler_feat_900.bin")
# MCMC_params = scaler_feat.transform(MCMC_params)
# MCMC_params_up = scaler_feat.transform(MCMC_params_up)
# MCMC_params_low = scaler_feat.transform(MCMC_params_low)
# # Make predictions on the galform set
# yhat_all = load_all_models(n_models=5, X_test=MCMC_params)
# yhat_avg = np.mean(yhat_all, axis=0)
# yhat_avg = yhat_avg[0]
# yhatz = yhat_avg[0:13]
# yhatk = yhat_avg[13:22]
#
# yhat_all_up = load_all_models(n_models=5, X_test=MCMC_params_up)
# yhat_avg_up = np.mean(yhat_all_up, axis=0)
# yhat_avg_up = yhat_avg_up[0]
# yhatz_up = yhat_avg_up[0:13]
# yhatk_up = yhat_avg_up[13:22]
#
# yhat_all_low = load_all_models(n_models=5, X_test=MCMC_params_low)
# yhat_avg_low = np.mean(yhat_all_low, axis=0)
# yhat_avg_low = yhat_avg_low[0]
# yhatz_low = yhat_avg_low[0:13]
# yhatk_low = yhat_avg_low[13:22]
#
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# axs.plot(bins[0:13], yhatz, 'b--', label="MCMC fit")
# axs.plot(bins[0:13], yhatz_up, '--', label="MCMC fit upper bound", alpha=0.5)
# axs.plot(bins[0:13], yhatz_low, '--', label="MCMC fit lower bound", alpha=0.5)
# dflc.plot(ax=axs, x="z", y="dN(>S)/dz", color='blue', label="Lacey et al. 2016")
# axs.set_ylabel(r"Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
# axs.set_xlabel(r"Redshift, z", fontsize=15)
# plt.tick_params(labelsize=15)
# plt.legend()
# plt.show()
#
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# axs.plot(bins[13:22], yhatk, 'b--', label="MCMC fit")
# axs.plot(bins[13:22], yhatk_up, '--', label="MCMC fit upper bound", alpha=0.5)
# axs.plot(bins[13:22], yhatk_low, '--', label="MCMC fit lower bound", alpha=0.5)
# axs.plot(binsk_full, k_test_lc_full, 'b-', label="Lacey et al. 2016")
# axs.set_xlabel(r"M$_{AB}$ - 5log(h)", fontsize=15)
# axs.set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)", fontsize=15)
# plt.tick_params(labelsize=15)
# plt.legend()
# axs.set_xlim(-18, -25)
# axs.set_ylim(-6, -1)
# plt.show()

# Save Lacey y values
# y_true = np.hstack([z_test_lc, k_test_lc_sub])
# np.save('Lacey_y_true.npy', y_true)
