import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from joblib import load


def masked_mae(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)  # Create a mask where non-zero values are True
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))

    return loss

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

def load_all_models(n_models, X_test, sub_no):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
           X_test: test sample in np.array already normalized
           sub_no: model subsample number, either 900, 600 or 300
    :return: list of ensemble models
    """

    all_yhat = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_2512_mask_' + str(sub_no)
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={"masked_mae": masked_mae}, compile=False)
        print('>loaded %s' % filename)
        # Produce prediction
        yhat = model.predict(X_test)
        all_yhat.append(yhat)

    return all_yhat

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
# feature_file = 'Data/Data_for_ML/testing_data/feature'
# label_file = 'Data/Data_for_ML/testing_data/label_sub12_dndz_fix'

# X_test = genfromtxt(feature_file)
# y_test = genfromtxt(label_file)

X_test = np.load('Data/Data_for_ML/testing_data/X_test_100.npy')
y_test = np.load('Data/Data_for_ML/testing_data/y_test_100.npy')

scaler_feat = load('mm_scaler_feat.bin')
X_test = scaler_feat.transform(X_test)

# Load scalar fits
# scaler_feat = MinMaxScaler(feature_range=(0, 1))
# scaler_feat.fit(X_test)
# X_test = scaler_feat.transform(X_test)
# Use standard scalar for the label data
# scaler_label = StandardScaler()
# scaler_label.fit(y_test)
# y_test = scaler_label.transform(y_test)
#
# model_9 = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask_900', compile=False)
# model_6 = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask_600', compile=False)
# model_3 = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask_300', compile=False)
#
# yhat_9 = model_9.predict(X_test)
# yhat_6 = model_6.predict(X_test)
# yhat_3 = model_3.predict(X_test)

yhat_all_900 = load_all_models(n_models=5, X_test=X_test, sub_no=900)
yhat_avg_900 = np.mean(yhat_all_900, axis=0)
yhat_all_600 = load_all_models(n_models=5, X_test=X_test, sub_no=600)
yhat_avg_600 = np.mean(yhat_all_600, axis=0)
yhat_all_300 = load_all_models(n_models=5, X_test=X_test, sub_no=300)
yhat_avg_300 = np.mean(yhat_all_300, axis=0)

# yhat_2 = scaler_label.inverse_transform(yhat_2)
# yhat_4 = scaler_label.inverse_transform(yhat_4)
# yhat_6 = scaler_label.inverse_transform(yhat_6)

# y_test = scaler_label.inverse_transform(y_test)

# Import the counts bins x axis
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#fig.subplots_adjust(wspace=0)

axs[0,0].plot(bins[0:13], yhat_avg_900[7][0:13], '--', label="Avg 900 model")
axs[0,0].plot(bins[0:13], yhat_avg_600[7][0:13], '--', label="Avg 600 model")
axs[0,0].plot(bins[0:13], yhat_avg_300[7][0:13], '--', label="Avg 300 model")
axs[0,0].plot(bins[0:13], y_test[7][0:13], 'gx-', label="True model")
axs[0,0].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
axs[0,0].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
axs[0,0].legend()
axs[0,0].set_xlabel("Redshift, z", fontsize=16)
axs[0,0].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[0,1].plot(bins[0:13], yhat_avg_900[72][0:13], '--', label="Avg 900 model")
axs[0,1].plot(bins[0:13], yhat_avg_600[72][0:13], '--', label="Avg 600 model")
axs[0,1].plot(bins[0:13], yhat_avg_300[72][0:13], '--', label="Avg 300 model")
axs[0,1].plot(bins[0:13], y_test[72][0:13], 'gx-', label="True model")
axs[0,1].errorbar(bag_df["x"], bag_df["y"], yerr=(lower_bag, upper_bag), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Bagley et al. 2020')
axs[0,1].plot(bag_df["x"].iloc[-2], bag_df["y"].iloc[-2], 'wo', markeredgecolor='black', zorder=3)
axs[0,1].legend()
axs[0,1].set_xlabel("Redshift, z", fontsize=16)
axs[0,1].set_ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]', fontsize=16)

axs[1,0].plot(bins[13:22], yhat_avg_900[7][13:22], '--', label="Avg 900 model")
axs[1,0].plot(bins[13:22], yhat_avg_600[7][13:22], '--', label="Avg 600 model")
axs[1,0].plot(bins[13:22], yhat_avg_300[7][13:22], '--', label="Avg 300 model")
axs[1,0].plot(bins[13:22], y_test[7][13:22], 'gx-', label="True model")
axs[1,0].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1,0].legend()
axs[1,0].invert_xaxis()
axs[1,0].set_xlabel("K-band magnitude", fontsize=16)
axs[1,0].set_ylim((-6, -1))
axs[1,0].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)

axs[1,1].plot(bins[13:22], yhat_avg_900[72][13:22], '--', label="Avg 900 model")
axs[1,1].plot(bins[13:22], yhat_avg_600[72][13:22], '--', label="Avg 600 model")
axs[1,1].plot(bins[13:22], yhat_avg_300[72][13:22], '--', label="Avg 300 model")
axs[1,1].plot(bins[13:22], y_test[72][13:22], 'gx-', label="True model")
axs[1,1].errorbar(drive_df["Mag"], drive_df["LF"], yerr=(lower_dri, upper_dri), markeredgecolor='black',
                  ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
axs[1,1].legend()
axs[1,1].invert_xaxis()
axs[1,1].set_xlabel("K-band magnitude", fontsize=16)
axs[1,1].set_ylim((-6, -1))
axs[1,1].set_ylabel(r'Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s]', fontsize=16)
plt.show()

X = [300, 600, 900]
MAEC_single = [0.0886, 0.0587, 0.0505]
MAEC_avg = [0.0805, 0.0578, 0.0495]

MAEz_single = [0.0715, 0.0420, 0.0343]
MAEz_avg = [0.0645, 0.0410, 0.0329]

MAEk_single = [0.1057, 0.0755, 0.0667]
MAEk_avg = [0.0966, 0.0746, 0.0660]

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(6, 11),
                        facecolor='w', edgecolor='k', sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].plot(X, MAEC_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[0].plot(X, MAEC_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[0].set_ylabel(r'MAE$_{combo}$')
axs[0].legend()

axs[1].plot(X, MAEz_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[1].plot(X, MAEz_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[1].set_ylabel(r'MAE$_{dn/dz}$')
axs[1].legend()

axs[2].plot(X, MAEk_single, linestyle='dotted', marker='o', color='tab:blue', label="Single fixed model")
axs[2].plot(X, MAEk_avg, linestyle='dashed', marker='o', color='tab:blue', label="Averaged fixed ensemble model")
axs[2].set_ylabel(r'MAE$_{k-band}$')
axs[2].legend()

axs[2].set_xlabel('Training examples')

plt.show()
