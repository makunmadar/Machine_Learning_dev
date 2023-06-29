# Predicting the plots from Lacey et al. 16 and Elliott et al. 21
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from numpy import genfromtxt
from joblib import load


def load_all_models(n_models):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_1000_S'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()},
                                           compile=False)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models

def emline_df(path, columns):
    data = []
    with open(path,'r') as fh:
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
    #df['Mag'] = np.log10(df['Mag'].replace(0, np.nan))
    df['Krdust'] = np.log10(df['Krdust'].replace(0, 1e-20))

    return df

#  Lacey parameters
X_lacey = np.array([1.0, 320, 320, 3.4, 0.8, 0.74])
X_elliot = np.array([0.59, 489, 284, 2.24, 0.84, 0.2])
X_test = np.vstack((X_lacey, X_elliot))

# Load scalar fits
# scaler_feat = MinMaxScaler(feature_range=(0,1))
# scaler_feat.fit(X_test)
# X_test = scaler_feat.transform(X_test)
scaler_feat = load("mm_scaler_feat.bin")
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = load("std_scaler_label.bin")
print(scaler_label)

# Load a model from the Model directory
model_1 = tf.keras.models.load_model('Models/Ensemble_model_1_1000_S', compile=False)
model_2 = tf.keras.models.load_model('Models/Ensemble_model_2_1000_S', compile=False)

n_members = 2
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Make a prediction for test data
yhat_1 = model_1.predict(X_test)
yhat_2 = model_2.predict(X_test)

# ensamble_pred = list()
# for model in members:
#     yhat = model.predict(X_test)
#     yhat = scaler_label.inverse_transform(yhat)
#     ensamble_pred.append(yhat)
#
# yhatavg = np.mean(ensamble_pred, axis=0)

# De-normalize the predictions and truth data
yhat_1 = scaler_label.inverse_transform(yhat_1)
yhat_2 = scaler_label.inverse_transform(yhat_2)

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

path = "Data/Data_for_ML/Observational/Lacey_16/dndz_Bagley_HaNII_ext"
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

with open(path,'r') as fh:
     for curline in fh:
        if curline.startswith("#"):
            header = curline
        else:
            break

header = header[1:].strip().split()

dflc = pd.DataFrame(data=data, columns=header)
dflc = dflc.apply(pd.to_numeric)

path_ed = "Data/Data_for_ML/Observational/Elliott_21/dndz_Bagley_HaNII_ext"
flist_ed = open(path_ed).readlines()
data_ed = []
parsing = False
for line in flist_ed:
    if line.startswith("# S_nu/Jy=   2.1049E+07"):
        parsing = True
    elif line.startswith("# S_nu/Jy=   2.3645E+07"):
        parsing = False
    if parsing:
        if line.startswith("#"):
            header_ed = line
        else:
            row = line.strip().split()
            data_ed.append(row)

data_ed = np.vstack(data_ed)

df_ed = pd.DataFrame(data=data_ed, columns = header)
df_ed = df_ed.apply(pd.to_numeric)
#
dflc['dN(>S)/dz'] = np.log10(dflc['dN(>S)/dz'].replace(0, np.nan))
dflc = dflc[dflc['z']<=1.6007]
dflc = dflc[dflc['z']>=0.9]
df_ed['dN(>S)/dz'] = np.log10(df_ed['dN(>S)/dz'].replace(0, np.nan))
df_ed = df_ed[df_ed['z']<=1.6007]
df_ed = df_ed[df_ed['z']>=0.9]
#
fig, axs = plt.subplots(1, 1, figsize=(13,8))

axs.plot(bins[0:13], yhat_1[0][0:13], 'b--')
axs.plot(bins[0:13], yhat_2[0][0:13], 'b--', alpha=0.5)
# axs.plot(bins[0:13], yhatavg[0][0:13], 'b--', label="Avg ensemble LC")
axs.plot(bins[0:13], yhat_1[1][0:13], 'g--', alpha=0.5)
axs.plot(bins[0:13], yhat_2[1][0:13], 'g--', alpha=0.5)
# axs.plot(bins[0:13], yhatavg[1][0:13], 'g--', label="Avg ensemble ED")

# Original galform data
dflc.plot(ax = axs, x="z", y="dN(>S)/dz", label = "Lacey et al. 2016", color = 'blue')
df_ed.plot(ax = axs, x="z", y="dN(>S)/dz", label = "Elliott et al. 2021", color = "green")
axs.set_ylabel(r"Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize = 15)
axs.set_xlabel(r"Redshift, z", fontsize = 15)
axs.set_xlim(0.9, 1.6)
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
path_lfed = "Data/Data_for_ML/Observational/Elliott_21/gal.lf"

df_lflc = emline_df(path_lflc, columns_t)
df_lfed = emline_df(path_lfed, columns_t)

fig, axs = plt.subplots(1, 1, figsize=(13,8))

axs.plot(bins[13:22], yhat_1[0][13:22], 'b--')
axs.plot(bins[13:22], yhat_2[0][13:22], 'b--', alpha=0.5)
# axs.plot(bins[13:22], yhatavg[0][13:22], 'b--', label="Avg ensemble LC")
axs.plot(bins[13:22], yhat_1[1][13:22], 'g--', alpha=0.5)
axs.plot(bins[13:22], yhat_2[1][13:22], 'g--', alpha=0.5)
# axs.plot(bins[13:22], yhatavg[1][13:22], 'g--', label="Avg ensemble ED")

df_lflc.plot(ax=axs, x="Mag", y="Krdust", legend=True, color='blue', label="Lacey et al. 2016")
df_lfed.plot(ax=axs, x="Mag", y="Krdust", legend=True, color='green', label="Elliott et al. 2021")

axs.set_xlabel(r"M$_{AB}$ - 5log(h)", fontsize=15)
axs.set_ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)", fontsize=15)
axs.set_xlim(-18, -25)
axs.set_ylim(-6, -1)
plt.tick_params(labelsize=15)
plt.show()
