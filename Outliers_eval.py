import numpy as np
from numpy import genfromtxt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


def kband_df(path, columns):
    '''
    This function extracts just the k_band LF data and saves it in a dataframe.

    :param path: path to the LF file
    :param columns: what are the names of the magnitude columns?
    :return: dataframe
    '''
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

    return df

# Make a prediction with the stacked model
def predict_stacked_model(model, inputX):
    # Prepare input data
    X = [inputX for _ in range(len(model.input))]
    # Make prediction
    return model.predict(X)

# Import the observational data
# Import the Bagley et al. 2020 data from .csv file
bag_df = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Corrected_Ha_Bagley_redshift.csv", delimiter=',')
upper_bag = np.log10(bag_df["+"]) - np.log10(bag_df["y"])
lower_bag = np.log10(bag_df["y"]) - np.log10(bag_df["-"])
bag_df["y"] = np.log10(bag_df["y"])

# Import the Driver et al. 2012 data from .data file
columns_d = ['Mag', 'LF', 'error', 'Freq']
drive_df = kband_df("Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data", columns_d)
drive_df = drive_df[(drive_df != 0).all(1)]
drive_df['LF'] = drive_df['LF']*2 # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
drive_df['error'] = drive_df['error']*2
upper_dri = np.log10(drive_df['LF'] + drive_df['error']) - np.log10(drive_df['LF'])
lower_dri = np.log10(drive_df['LF']) - np.log10(drive_df['LF'] - drive_df['error'])
drive_df['LF'] = np.log10(drive_df['LF'])

# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz_fix'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test = scaler_label.transform(y_test)

model_2 = tf.keras.models.load_model('Models/stacked_model_200', compile=False)
model_4 = tf.keras.models.load_model('Models/stacked_model_400', compile=False)
model_6 = tf.keras.models.load_model('Models/stacked_model_600', compile=False)
model_8 = tf.keras.models.load_model('Models/stacked_model_800', compile=False)
model_10 = tf.keras.models.load_model('Models/stacked_model_1000', compile=False)

yhat_2 = predict_stacked_model(model_2, X_test)
yhat_4 = predict_stacked_model(model_4, X_test)
yhat_6 = predict_stacked_model(model_6, X_test)
yhat_8 = predict_stacked_model(model_8, X_test)
yhat_10 = predict_stacked_model(model_10, X_test)

yhat_2 = scaler_label.inverse_transform(yhat_2)
yhat_4 = scaler_label.inverse_transform(yhat_4)
yhat_6 = scaler_label.inverse_transform(yhat_6)
yhat_8 = scaler_label.inverse_transform(yhat_8)
yhat_10 = scaler_label.inverse_transform(yhat_10)

y_test = scaler_label.inverse_transform(y_test)

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#fig.subplots_adjust(wspace=0)

axs[0,0].plot(bins[0:12], yhat_2[3][0:12], '--', label="200 model")
axs[0,0].plot(bins[0:12], yhat_4[3][0:12], '--', label="400 model")
axs[0,0].plot(bins[0:12], yhat_6[3][0:12], '--', label="600 model")
axs[0,0].plot(bins[0:12], yhat_8[3][0:12], '--', label="800 model")
axs[0,0].plot(bins[0:12], yhat_10[3][0:12], '--', label="1000 model")
axs[0,0].plot(bins[0:12], y_test[3][0:12], 'gx-', label="True model")
axs[0,0].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
axs[0,0].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
axs[0,0].legend()
axs[0,0].set_xlabel("Redshift, z", fontsize=16)
axs[0,0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[0,1].plot(bins[0:12], yhat_2[7][0:12], '--', label="200 model")
axs[0,1].plot(bins[0:12], yhat_4[7][0:12], '--', label="400 model")
axs[0,1].plot(bins[0:12], yhat_6[7][0:12], '--', label="600 model")
axs[0,1].plot(bins[0:12], yhat_8[7][0:12], '--', label="800 model")
axs[0,1].plot(bins[0:12], yhat_10[7][0:12], '--', label="1000 model")
axs[0,1].plot(bins[0:12], y_test[7][0:12], 'gx-', label="True model")
axs[0,1].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
axs[0,1].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
axs[0,1].legend()
axs[0,1].set_xlabel("Redshift, z", fontsize=16)
axs[0,1].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[1,0].plot(bins[12:24], yhat_2[3][12:24], '--', label="200 model")
axs[1,0].plot(bins[12:24], yhat_4[3][12:24], '--', label="400 model")
axs[1,0].plot(bins[12:24], yhat_6[3][12:24], '--', label="600 model")
axs[1,0].plot(bins[12:24], yhat_8[3][12:24], '--', label="800 model")
axs[1,0].plot(bins[12:24], yhat_10[3][12:24], '--', label="1000 model")
axs[1,0].plot(bins[12:24], y_test[3][12:24], 'gx-', label="True model")
axs[1,0].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1,0].legend()
axs[1,0].invert_xaxis()
axs[1,0].set_xlabel("K-band magnitude", fontsize=16)
axs[1,0].set_ylim((-6, -1))
axs[1,0].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)

axs[1,1].plot(bins[12:24], yhat_2[7][12:24], '--', label="200 model")
axs[1,1].plot(bins[12:24], yhat_4[7][12:24], '--', label="400 model")
axs[1,1].plot(bins[12:24], yhat_6[7][12:24], '--', label="600 model")
axs[1,1].plot(bins[12:24], yhat_8[7][12:24], '--', label="800 model")
axs[1,1].plot(bins[12:24], yhat_10[7][12:24], '--', label="1000 model")
axs[1,1].plot(bins[12:24], y_test[7][12:24], 'gx-', label="True model")
axs[1,1].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1,1].legend()
axs[1,1].invert_xaxis()
axs[1,1].set_xlabel("K-band magnitude", fontsize=16)
axs[1,1].set_ylim((-6, -1))
axs[1,1].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

X = [200, 400, 600, 800, 1000]
MAEC_single = [0.106, 0.087, 0.062, 0.052, 0.0516]
MAEC_avg = [0.099, 0.082, 0.057, 0.048, 0.0479]
MAEC_stack = [0.090, 0.063, 0.055, 0.052, 0.0496]

MAEz_single = [0.096, 0.081, 0.0576, 0.0500, 0.0485]
MAEz_avg = [0.0917, 0.076, 0.0536, 0.0489, 0.0467]
MAEz_stack = [0.0846, 0.0578, 0.0521, 0.0488, 0.0469]

MAEk_single = [0.115, 0.0931, 0.0665, 0.0541, 0.0546]
MAEk_avg = [0.107, 0.087, 0.061, 0.0481, 0.0492]
MAEk_stack = [0.0959, 0.0673, 0.0578, 0.0537, 0.0523]

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(6, 11),
                        facecolor='w', edgecolor='k', sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].plot(X, MAEC_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[0].plot(X, MAEC_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[0].plot(X, MAEC_stack, linestyle='solid', marker='o', color='tab:blue', label="Stacked fixed ensemble model")
axs[0].set_ylabel(r'MAE$_{combo}$')
axs[0].legend()

axs[1].plot(X, MAEz_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[1].plot(X, MAEz_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[1].plot(X, MAEz_stack, linestyle='solid', marker='o', color='tab:blue', label="Stacked fixed ensemble model")
axs[1].set_ylabel(r'MAE$_{dn/dz}$')
axs[1].legend()

axs[2].plot(X, MAEk_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[2].plot(X, MAEk_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[2].plot(X, MAEk_stack, linestyle='solid', marker='o', color='tab:blue', label="Stacked fixed ensemble model")
axs[2].set_ylabel(r'MAE$_{k-band}$')
axs[2].legend()

axs[2].set_xlabel('Training examples')

plt.show()
