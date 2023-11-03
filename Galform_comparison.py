# Predicting the plots from Lacey et al. 16
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error
from Loading_functions import predict_all_models, lf_df, dndz_df
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

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

    df = df[(df['Mag'] <= -16.00)]
    df = df[(df['Mag'] >= -24.00)]
    df.reset_index(drop=True, inplace=True)
    # df['Mag'] = np.log10(df['Mag'].replace(0, np.nan))
    # df['Krdust'] = np.log10(df['Krdust'].replace(0, 1e-20))
    df['Krdust'] = np.log10(df['Krdust'].mask(df['Krdust'] <=0)).fillna(0)
    df['Rrdust'] = np.log10(df['Rrdust'].mask(df['Rrdust'] <=0)).fillna(0)

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
X_test = np.array([1.0, 320, 320, 3.4, 0.8, 0.74, 0.9, 0.3, 0.05, 0.005, 0.1])
X_test_old = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])

X_test = X_test.reshape(1, -1)

# Make predictions on the galform set
yhat_all = predict_all_models(n_models=5, X_test=X_test, variant='_6x5_mask_2899_LRELU_int')
yhat_avg = np.mean(yhat_all, axis=0)
# yhat_all_old = predict_all_models(n_models=1, X_test=X_test, variant='_6x5_mask_1000_LRELU_int')
# yhat_avg_old = np.mean(yhat_all_old, axis=0)
# yhat_all_old1 = predict_all_models(n_models=1, X_test=X_test, variant='_6x5_mask_2899_LRELU_int')
# yhat_avg_old1 = np.mean(yhat_all_old1, axis=0)

yhatz = yhat_avg[0][0:7]
# yhatz_old = yhat_avg_old[0][0:7]
# yhatz_old1 = yhat_avg_old1[0][0:7]
yhatk = yhat_avg[0][7:25]
# yhatk_old = yhat_avg_old[0][7:25]
# yhatk_old1 = yhat_avg_old1[0][7:25]
yhatr = yhat_avg[0][25:45]
# yhatr_old = yhat_avg_old[0][25:45]
# yhatr_old1 = yhat_avg_old1[0][25:45]

# Predict using the old set and old model here...

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_fullup_int'
bins = genfromtxt(bin_file)
bins_l = np.load('Lacey_bins.npy')

# Redshift distribution
path_zlc = "Data/Data_for_ML/Observational/Lacey_16/dndz_Bagley_HaNII_ext"
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
dflc = dndz_df(path_zlc, columns_Z)

z_test_lc = dflc['dN(>S)/dz'].values

# Manual MAE score
# maelc_z = mean_absolute_error(z_test_lc, yhatz)
#
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
#
# axs.plot(bins[0:7], yhatz, 'b--', label=f"Prediction")  # MAE: {maelc_z:.3f}")
#
# # Original galform data
# dflc.plot(ax=axs, x="z", y="dN(>S)/dz", color='blue', label="Lacey et al. 2016")
# # axs.scatter(bins[0:49], z_test_lc, color='blue', marker='x', label="Evaluation bins")
# axs.set_ylabel(r"log$_{10}$(dN(>S)/dz) [deg$^{-2}$]")
# axs.set_xlabel(r"Redshift, z")
# axs.set_xlim(0.9, 1.6)
# axs.set_ylim(3.1, 3.8)
# plt.legend()
# plt.show()

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

# K-band LF
k_test_lc_full = df_lflc['Krdust'].values
k_test_lc_sub = k_test_lc_full

# Ignore the zero truth values
# yhatk_lc_sub = yhatk[k_test_lc_sub != 0]
# binsk_lc_sub = bins[49:67][k_test_lc_sub != 0]
# k_test_lc_sub = k_test_lc_sub[k_test_lc_sub != 0]

binsk_full = df_lflc['Mag'][k_test_lc_full != 0]
k_test_lc_full = k_test_lc_full[k_test_lc_full != 0]

# Manual MAE score
# maelc_k = mean_absolute_error(k_test_lc_sub, yhatk_lc_sub)

# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
#
# axs.plot(bins[7:25], yhatk, 'b--', label=f"Prediction")  # MAE: {maelc_k:.3f}")
#
# axs.plot(binsk_full, k_test_lc_full, 'b-', label="Lacey et al. 2016")
# # axs.scatter(binsk_lc_sub, k_test_lc_sub, color='blue', marker='x', label="Evaluation bins")
# axs.set_xlabel(r"M$_{K,AB}$ - 5log(h)")
# axs.set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
# axs.set_xlim(-16, -24.5)
# axs.set_ylim(-5.5, -1)
# plt.legend()
# plt.show()

# R-band LF
r_test_lc_full = df_lflc['Rrdust'].values
r_test_lc_sub = r_test_lc_full

# Ignore the zero truth values
# yhatr_lc_sub = yhatr[r_test_lc_sub != 0]
# binsr_lc_sub = bins[49:67][r_test_lc_sub != 0]
# r_test_lc_sub = r_test_lc_sub[r_test_lc_sub != 0]

binsr_full = df_lflc['Mag'][r_test_lc_full != 0]
r_test_lc_full = r_test_lc_full[r_test_lc_full != 0]

# Manual MAE score
# maelc_r = mean_absolute_error(r_test_lc_sub, yhatr_lc_sub)

# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
#
# axs.plot(bins[25:45], yhatr, 'b--', label=f"Prediction")  # MAE: {maelc_r:.3f}")
# axs.plot(binsr_full, r_test_lc_full, 'b-', label="Lacey et al. 2016")
# # axs.scatter(binsk_lc_sub, k_test_lc_sub, color='blue', marker='x', label="Evaluation bins")
# axs.set_xlabel(r"M$_{r,AB}$ - 5log(h)")
# axs.set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
# axs.set_xlim(-16, -24)
# axs.set_ylim(-5.5, -1)
# plt.legend()
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(22, 6))
axs[0].plot(bins[0:7], yhatz, 'b--', label=f"Prediction")  # MAE: {maelc_z:.3f}")
# axs[0].plot(bins[0:7], yhatz_old, 'r--', label=f"1000 prediction")  # MAE: {maelc_z:.3f}")
# axs[0].plot(bins[0:7], yhatz_old1, 'g--', label=f"2899 prediction")  # MAE: {maelc_z:.3f}")

# dflc.plot(ax=axs[0], x="z", y="dN(>S)/dz", color='blue', label="Lacey et al. 2016")
axs[0].plot(dflc['z'], dflc['dN(>S)/dz'], 'b-', label="Lacey et al. 2016")
# axs[0].scatter(bins[0:7], z_test_lc, color='blue', marker='x', label="Evaluation bins")
axs[0].set_ylabel(r"log$_{10}$(dN(>S)/dz) [deg$^{-2}$]")
axs[0].set_xlabel(r"Redshift, z")
axs[0].set_xlim(0.9, 1.6)
axs[0].set_ylim(3.1, 3.8)

axs[1].plot(bins[7:25], yhatk, 'b--', label=f"Prediction")  # MAE: {maelc_k:.3f}")
# axs[1].plot(bins[7:25], yhatk_old, 'r--', label=f"1000 prediction")  # MAE: {maelc_k:.3f}")
# axs[1].plot(bins[7:25], yhatk_old1, 'g--', label=f"2899 prediction")  # MAE: {maelc_k:.3f}")

axs[1].plot(binsk_full, k_test_lc_full, 'b-', label="Lacey et al. 2016")
# axs[2].scatter(binsk_lc_sub, k_test_lc_sub, color='blue', marker='x', label="Evaluation bins")
axs[1].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[1].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[1].set_xlim(-16, -24.5)
axs[1].set_ylim(-5, -1)

axs[2].plot(bins[7:25], yhatk, 'b--', label=f"Prediction")  # MAE: {maelc_k:.3f}")
# axs[2].plot(bins[7:25], yhatk_old, 'r--', label=f"1000 prediction")  # MAE: {maelc_k:.3f}")
# axs[2].plot(bins[7:25], yhatk_old1, 'g--', label=f"2899 prediction")  # MAE: {maelc_k:.3f}")

axs[2].plot(binsk_full, k_test_lc_full, 'b-', label="Lacey et al. 2016")
# axs[2].scatter(binsk_lc_sub, k_test_lc_sub, color='blue', marker='x', label="Evaluation bins")
axs[2].set_xlabel(r"M$_{K,AB}$ - 5log(h)")
axs[2].set_ylabel(r"log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)")
axs[2].set_xlim(-16, -24.5)
axs[2].set_ylim(-5, -1)

plt.legend()
# plt.savefig('Lacey2016_galform_comparison.pdf')
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
df_lck = lf_df(path_lflc, columns_t, mag_high=-15.25, mag_low=-24)
df_lck['Krdust'] = np.log10(df_lck['Krdust'].mask(df_lck['Krdust'] <=0)).fillna(0)
k_test_lc_full = df_lck['Krdust'].values
binsk_full = df_lck['Mag'].values

df_lcr = lf_df(path_lflc, columns_t, mag_high=-13.75, mag_low=-24)
df_lcr['Rrdust'] = np.log10(df_lcr['Rrdust'].mask(df_lcr['Rrdust'] <=0)).fillna(0)
r_test_lc_sub = df_lcr['Rrdust'].values
binsr_full = df_lcr['Mag'].values
binsr_full = binsr_full[r_test_lc_sub != 0]
r_test_lc_sub = r_test_lc_sub[r_test_lc_sub != 0]

y_true = np.hstack([z_test_lc, k_test_lc_full, r_test_lc_sub])
np.save('Lacey_y_true.npy', y_true)
l_bins = np.hstack([bins_l[0:49], binsk_full, binsr_full])
np.save('Lacey_bins.npy', l_bins)
