"""Testing the interpolation code between the galform predictions and the observables for MAE calculations"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import tensorflow as tf
from joblib import load
from sklearn.metrics import mean_absolute_error


def masked_mae(y_true, y_pred):
    # The tensorflow models custom metric, this won't affect the predictions
    # But it gets rid of the warning message
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

    df = df[(df['Mag'] <= -17.67)]
    df = df[(df['Mag'] >= -25.11)]
    return df


# Import the Bagley et al. 2020 data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
Ha_b["n"] = np.log10(Ha_b["n"])
Ha_b["+"] = np.log10(Ha_b["+"])
Ha_b["-"] = np.log10(Ha_b["-"])
Ha_ytop = Ha_b["+"] - Ha_b["n"]
Ha_ybot = Ha_b["n"] - Ha_b["-"]
sigma = (Ha_ytop + Ha_ybot) / 2

# Import one example from the testing set:
# X_all = np.load('Data/Data_for_ML/testing_data/X_test.npy')
# y_all = np.load('Data/Data_for_ML/testing_data/y_test.npy')
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
# X = X_all[1]
# y = y_all[1]
bins = genfromtxt(bin_file)

# Test with a random prediction
# An example of where the models thinks there is actually an increase in the LF at the bright end.
X_rand = np.array([1.44501626e+00, 5.28109086e+02, 4.29897441e+02, 2.84023718e+00, -2.99195420e-01, 6.75682679e-01])
X_rand = X_rand.reshape(1, -1)
scaler_feat = load("mm_scaler_feat.bin")
X_rand = scaler_feat.transform(X_rand)
model = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask',
                                   custom_objects={"masked_mae": masked_mae}, compile=False)
y = model.predict(X_rand)
y = y[0]

# Redshift distribution
# Perform interpolation
xz1 = bins[0:13]
yz1 = y[0:13]
xz2 = Ha_b['z'].values
yz2 = Ha_b['n'].values

# Interpolate or resample the daa onto the common x-axis
interp_funcz = interp1d(xz1, yz1, kind='linear', fill_value='extrapolate')
interp_yz1 = interp_funcz(xz2)

# Plot to see how this looks
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
             fmt='co', label=r"Bagley'20 Observed")
axs.plot(bins[0:13], y[0:13], 'b--', label="Test galform")
axs.plot(xz2, interp_yz1, 'bx', label='Interpolated galform')
plt.legend()
plt.show()

# Working out the MAE values
# error_weightsz = 1/sigma
weighted_maez = mean_absolute_error(yz2, interp_yz1)  # *error_weightsz
print("Weighted MAE redshift distribution: ", weighted_maez)

# Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = kband_df(drive_path, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

# Perform interpolation
xk1 = bins[13:22]
yk1 = y[13:22]
xk2 = df_k['Mag'].values
yk2 = df_k['LF'].values

# Interpolate or resample the data onto the common x-axis
interp_funck = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
interp_yk1 = interp_funck(xk2)

# Plot to see how this looks
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
df_k.plot(ax=axs, x="Mag", y="LF", label='Driver et al. 2012', yerr=[df_k['error_lower'], df_k['error_upper']],
          markeredgecolor='black', ecolor="black", capsize=2, fmt='co')
axs.plot(bins[13:22], y[13:22], 'b--', label='Test galform')
axs.plot(xk2, interp_yk1, 'bx', label='Interpolated galform')
axs.invert_xaxis()
plt.legend()
plt.show()

# Working out the MAE values
# error_weightsk = 1/df_k['error']
weighted_maek = mean_absolute_error(yk2, interp_yk1)  # *error_weightsk
print("Weighted MAE Luminosity function: ", weighted_maek)

# Test combining the two interpolations for a combined MAE
# combine interpolated y values
interp_y1 = np.hstack([interp_yz1, interp_yk1])
y2 = np.hstack([yz2, yk2])

MAE = mean_absolute_error(y2, interp_y1)
print("Combined MAE: ", MAE)
