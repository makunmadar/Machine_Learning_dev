"""
Taking in the raw number counts and redshift distribution outputs and formatting them appropriately for
machine learning training and evaluation.

For now only saving the redshift distribution outputs for training.
"""
import numpy as np
import pandas as pd
import os
import re
from sklearn.utils import shuffle
from numpy import genfromtxt
from scipy.interpolate import interp1d
from Loading_functions import lf_df


def dndz_df(path, columns):
    """
    This function extracts the redshift distribution data and saves it in a dataframe.

    :param path: path to the distribution file
    :param columns: names of the redshift distribution columns
    :return: dataframe
    """

    data = []
    flist = open(path).readlines()
    parsing = False
    for line in flist:
        if line.startswith('# S_nu/Jy=   2.1049E+07'):
            parsing = True
        elif line.startswith('# S_nu/Jy=   2.3645E+07'):
            parsing = False
        if parsing:
            if line.startswith('#'):
                header = line
            else:
                row = line.strip().split()
                data.append(row)

    data = np.vstack(data)
    df = pd.DataFrame(data=data, columns=columns)
    df = df.apply(pd.to_numeric)

    # Don't want to include potential zero values at z=0.692
    df = df[(df['z'] < 2.1)]
    df = df[(df['z'] > 0.7)]

    df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].mask(df['dN(>S)/dz'] <= 0)).fillna(0)

    return df


def find_number(text, c):
    '''
    Identify the model number of a path string

    :param text: model name as string
    :param c: after what string symbol does the number reside
    :return: the model number
    '''

    return re.findall(r'%s(\d+)' % c, text)


def round_sigfigs(x):
    """
    Round a ndarray to 3 significant figures
    :param x: ndarray
    :return: rounded ndarray
    """
    return np.around(x, -int(np.floor(np.log10(abs(x)))) + 2)

# Load in the Observational data
# Import Bagley et al. 2020
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
sigmaz = (Ha_b["+"].values - Ha_b['-'].values) / 2

# Import the Driver et al. 2012 data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path_k = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_path_k, driv_headers, mag_low=-23.75, mag_high=-15.25)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
sigmak = df_k['error'].values

drive_path_r = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_path_r, driv_headers, mag_low=-23.25, mag_high=-13.75)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
sigmar = df_r['error'].values

sigma = np.hstack([sigmaz, sigmak, sigmar])
obs = np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values])
frac_sigma = sigma/obs

# Import Cole et al. 2001
# cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
# cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
# df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00-1.87, mag_high=-16.00-1.87)
# df_ck = df_ck[df_ck['PhiK'] != 0]
# df_ck = df_ck.sort_values(['Mag'], ascending=[True])
# df_ck['Mag'] = df_ck['Mag'] + 1.87
# np.save('fractional_sigma.npy', frac_sigma)

# Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_HaNIIext_1999/dndz_HaNII_ext/"

base_filenames = os.listdir(base_path_dndz)
base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_Hadndz = np.empty((0, 7))
for file in base_filenames:
    model_number = find_number(file, '.')
    df_z = dndz_df(base_path_dndz + file, columns_Z)

    interp_funcz = interp1d(df_z['z'].values, df_z['dN(>S)/dz'].values, kind='linear', fill_value='extrapolate')
    interp_yz1 = interp_funcz(Ha_b['z'].values)

    training_Hadndz = np.vstack([training_Hadndz, interp_yz1])

dndzbins = Ha_b['z'].values
print('Redshift distribution bins: ', dndzbins)
print('Example of dn/dz values: ', training_Hadndz[1000])

# LF
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

base_path_lf = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_kband_training/LF_1999/LF/"
basek_filenames = os.listdir(base_path_lf)
basek_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_lf = np.empty((0, 38))
for file in basek_filenames:
    model_number = find_number(file, '.')

    df_lfk = lf_df(base_path_lf + file, columns_lf, mag_low=-23.80, mag_high=-15.04)  # -23.80, -15.04 for driver
    df_lfk['Krdust'] = np.log10(df_lfk['Krdust'].mask(df_lfk['Krdust'] <= 0)).fillna(0)
    df_lfk = df_lfk[df_lfk['Krdust'] != 0]
    interp_funck = interp1d(df_lfk['Mag'].values, df_lfk['Krdust'].values, kind='linear', fill_value='extrapolate',
                            bounds_error=False)
    interp_yk1 = interp_funck(df_k['Mag'].values)
    interp_yk1[df_k['Mag'].values < min(df_lfk['Mag'].values)] = 0

    df_lfr = lf_df(base_path_lf + file, columns_lf, mag_low=-23.36, mag_high=-13.73)
    df_lfr['Rrdust'] = np.log10(df_lfr['Rrdust'].mask(df_lfr['Rrdust'] <= 0)).fillna(0)
    df_lfr = df_lfr[df_lfr['Rrdust'] != 0]
    interp_funcr = interp1d(df_lfr['Mag'].values, df_lfr['Rrdust'].values, kind='linear', fill_value='extrapolate',
                            bounds_error=False)
    interp_yr1 = interp_funcr(df_r['Mag'].values)
    interp_yr1[df_r['Mag'].values < min(df_lfk['Mag'].values)] = 0

    # Calculate the amount of padding needed for both arrays
    # paddingk = [(0, 18 - len(interp_yk1))]  # 18 for driver
    # paddingr = [(0, 20 - len(interp_yr1))]

    # Pad both arrays with zeros
    # padded_arrayk = np.pad(interp_yk1, paddingk, mode='constant', constant_values=0)
    # padded_arrayr = np.pad(interp_yr1, paddingr, mode='constant', constant_values=0)

    lf_vector = np.concatenate((interp_yk1, interp_yr1))
    training_lf = np.vstack([training_lf, lf_vector])

# lfbins = np.concatenate((df_k['Mag'].values, df_r['Mag'].values))
lfbins = np.concatenate((df_k['Mag'].values, df_r['Mag'].values))
print('LF distribution bins: ', lfbins)
print('Example of k-band LF values: ', training_lf[1000][0:18])
print('Example of r-band LF values: ', training_lf[1000][18:38])

# Combine the two data sets with the parameter data
combo_bins = np.hstack([dndzbins, lfbins])  # This data is not required for the machine learning
combo_labels = np.hstack([training_Hadndz, training_lf])
# combo_labels = combo_labels[:1999]  # As we only have the first 1340 for definite
print('Combo bins: ', combo_bins)
print('Example of combo labels: ', combo_labels[1000])

# testing_feature_file1 = 'Data/Data_for_ML/raw_features/test_parameters.csv'
# testing_features1 = genfromtxt(testing_feature_file1, delimiter=',', skip_header=1)
# # Import the second feature file not including the redshift, subvolume or model information
# testing_feature_file2 = 'Data/Data_for_ML/raw_features/test_parameters_extended_v3.csv'
# testing_features2 = genfromtxt(testing_feature_file2, delimiter=',', skip_header=1, usecols= range(6))
# # Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
# testing_features2 = testing_features2[::30]
# testing_features = np.vstack([testing_features1, testing_features2])
# testing_features = np.vectorize(round_sigfigs)(testing_features)
# combo_labels = np.round(combo_labels, decimals=3)
# print('Example of features: ', testing_features[113])

training_feature_file = 'Data/Data_for_ML/raw_features/updated_parameters_extended_2000v3.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(11))
# Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
training_features = training_features[::30]
training_features = np.delete(training_features, 1470, axis=0)  # As for now we don't have model 1471

training_features = np.vectorize(round_sigfigs)(training_features)
# combo_labels = np.round(combo_labels, decimals=3)
# print('Example of rounded combo labels: ', combo_labels[113])

# Shuffle the data properly
training_features, combo_labels = shuffle(training_features, combo_labels)

# Save the arrays aas a text file
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
testing_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/testing_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
np.savetxt(training_path + 'label_fullup_int', combo_labels, fmt='%.2f')
# np.savetxt(testing_path + 'label_full_r', combo_labels, fmt='%.2f')
np.savetxt(training_path + 'feature_up', training_features, fmt='%.2f')
# np.savetxt(testing_path + 'feature', testing_features, fmt='%.2f')
np.savetxt(bin_path + 'bin_fullup_int', combo_bins, fmt='%.2f')

# plt.plot(kbins, training_kband[0], 'rx')
# kbins = kbins[0::2]
# training_kband = training_kband[0][0::2]
# plt.plot(kbins, training_kband, 'gx')
# plt.show()