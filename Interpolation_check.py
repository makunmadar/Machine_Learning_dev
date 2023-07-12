"""Testing the interpolation code between the galform predictions and the observables for MAE calculations"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


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
X_all = np.load('Data/Data_for_ML/testing_data/X_test.npy')
y_all = np.load('Data/Data_for_ML/testing_data/y_test.npy')
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
X = X_all[1]
y = y_all[1]
bins = genfromtxt(bin_file)

# Perform interpolation
xz1 = bins[0:13]
yz1 = y[0:13]
xz2 = Ha_b['z'].values
yz2 = Ha_b['n'].values

# Create common x-axis
common_xz = np.linspace(min(min(xz1), min(xz2)), max(max(xz1), max(xz2)), num=50)

# Interpolate or resample the daa onto the common x-axis
interp_funcz1 = interp1d(xz1, yz1, kind='quadratic', fill_value='extrapolate')
interp_funcz2 = interp1d(xz2, yz2, kind='quadratic', fill_value='extrapolate')
interp_yz1 = interp_funcz1(common_xz)
interp_yz2 = interp_funcz2(common_xz)

# Plot to see how this looks
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor="black", capsize=2,
             fmt='co', label=r"Bagley'20 Observed")
axs.plot(bins[0:13], y[0:13], 'bx', label="Test galform")
axs.plot(common_xz, interp_yz1, 'b--', label='Interpolated galform')
axs.plot(common_xz, interp_yz2, 'g--', label='Interpolated bagley')
plt.legend()
plt.show()

# Working out the MAE values
# error_weightsz = 1/sigma
weighted_errorz = np.abs(interp_yz1-interp_yz2)# *error_weightsz
weighted_maez = np.mean(weighted_errorz)
print("Weighted MAE redshift distribution: ", weighted_maez)

# Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = kband_df(drive_path, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF']*2 # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error']*2 # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

# Perform interpolation
xk1 = bins[13:22]
yk1 = y[13:22]
xk2 = df_k['Mag'].values
yk2 = df_k['LF'].values

# Create common x-axis
common_xk = np.linspace(min(min(xk1), min(xk2)), max(max(xk1), max(xk2)), num=50)

# Interpolate or resample the daa onto the common x-axis
interp_funck1 = interp1d(xk1, yk1, kind='linear', fill_value='extrapolate')
interp_funck2 = interp1d(xk2, yk2, kind='linear', fill_value='extrapolate')
interp_yk1 = interp_funck1(common_xk)
interp_yk2 = interp_funck2(common_xk)

# Plot to see how this looks
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
df_k.plot(ax=axs, x="Mag", y="LF", label='Driver et al. 2012', yerr=[df_k['error_lower'], df_k['error_upper']],
          markeredgecolor='black', ecolor="black", capsize=2, fmt='co')
axs.plot(bins[13:22], y[13:22], 'bx', label='Test galform')
axs.plot(common_xk, interp_yk1, 'b--', label='Interpolated galform')
axs.plot(common_xk, interp_yk2, 'g--', label='Interpolated driver')
axs.invert_xaxis()
plt.legend()
plt.show()

# Working out the MAE values
# error_weightsk = 1/df_k['error']
weighted_errork = np.abs(interp_yk1-interp_yk2)# *error_weightsk
weighted_maek = np.mean(weighted_errork)
print("Weighted MAE Luminosity function: ", weighted_maek)

# 