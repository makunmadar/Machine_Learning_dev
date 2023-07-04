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
from scipy.optimize import curve_fit
from numpy import genfromtxt
import matplotlib.pyplot as plt


def lin(x, m, c):
    y = (m*x) + c
    y = np.exp(y)
    return y


def phi(M, Ps, Ms, b):
    """
    Schecter function for curve fitting tool, fixing the variable alpha as this only affects the faint end and
    this is to optimize the bright end.
    Setting alpha to roughly the local universe, -1.3.

    :param M: Variable x axis representing the absolute magnitude
    :param Ms: Schecter function parameter M*
    :param ps: Schecter function parameter phi*
    :param b: special parameter to limit the exponential factor
    :return: Schecter funtion phi(L)
    """

    a = -1.3
    #b = 0.45
    phi_L = (np.log(10) / 2.5) * Ps * ((10 ** (0.4 * (Ms - M))) ** (a + 1)) * np.exp(-(10 ** (0.4 * (Ms - M))) ** b)
    return phi_L


def kband_df(path, columns):
    """
    This function extracts just the k_band LF data and saves it in a dataframe.

    :param path: path to the LF file
    :param columns: what are the names of the magnitude columns?
    :return: dataframe
    """
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

    # Keep it between the magnitude range -18<Mag_k<-25
    df = df[(df['Mag'] <= -17.67)]
    df = df[(df['Mag'] >= -25.11)]
    df.reset_index(drop=True, inplace=True)

    # Replacing the zero values with a prediction from a schechter fitting.
    # See Testing_schec.py for development of this code.

    # kbins = df['Mag'].values

    # if (df['Krdust'] == 0).any():
    #
    #     y = df['Krdust'].values
    #     err = df['Krdust(error)'].values
    #
    #     zeroidx = [i for i, e in enumerate(y) if e == 0]
    #     x = kbins[y > 0]
    #     err = err[y > 0]
    #     y = y[y > 0]
    #
    #     x_sec = x[0:4]
    #     y_sec = y[0:4]
    #     err_sec = err[0:4]
    #
    #     #params, cov = curve_fit(phi, x_sec, y_sec, bounds=((0, -22, 0), (10, -15, 1)), sigma=err_sec)
    #     params, cov = curve_fit(lin, x_sec, y_sec, sigma=err_sec)
    #
    #     for i in zeroidx:
    #         #phi_p = phi(kbins[i], params[0], params[1], params[2])
    #         phi_p = lin(kbins[i], params[0], params[1])
    #         df['Krdust'].iloc[i] = phi_p
    #
    # df['Krdust'] = np.log10(df['Krdust'])
    df['Krdust'] = np.log10(df['Krdust'].mask(df['Krdust'] <=0)).fillna(0)

    #df['Krdust'] = np.log10(df['Krdust'].replace(0, 1E-20))
    # df['Krdust'] = np.log10(df['Krdust'].replace(0,  0.0000025119))

    return df


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

    # df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].replace(0, 1E-20))
    df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].mask(df['dN(>S)/dz'] <=0)).fillna(0)

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


# Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_training_1000/"

base_filenames = os.listdir(base_path_dndz)
base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_Hadndz = np.empty((0, 13))
for file in base_filenames:
    model_number = find_number(file, '.')
    df = dndz_df(base_path_dndz + file, columns_Z)

    dndz_vector = df['dN(>S)/dz'].values

    # Subsample by taking every 4th value
    dndz_vector = dndz_vector[0::4]

    training_Hadndz = np.vstack([training_Hadndz, dndz_vector])

dndzbins = df['z'].values
dndzbins = dndzbins[0::4]
print('Redshift distribution bins: ', dndzbins)
print('Example of dn/dz values: ', training_Hadndz[113])

# K-band LF
columns_k = ['Mag', 'Ur', 'Ur(error)', 'Urdust', 'Urdust(error)',
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

base_path_kband = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_kband_training/k_band_ext/"
basek_filenames = os.listdir(base_path_kband)
basek_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_kband = np.empty((0, 9))

for file in basek_filenames:
    model_number = find_number(file, '.')
    df_k = kband_df(base_path_kband + file, columns_k)

    k_vector = df_k['Krdust'].values

    # Subsample by taking every other value
    k_vector = k_vector[0::2]
    training_kband = np.vstack([training_kband, k_vector])

kbins = df_k['Mag'].values
kbins = kbins[0::2]
print('k-band LF distribution bins: ', kbins)
print('Example of k-band LF values: ', training_kband[113])

# Combine the two data sets with the parameter data
combo_bins = np.hstack([dndzbins, kbins])  # This data is not required for the machine learning
combo_labels = np.hstack([training_Hadndz, training_kband])
print('Combo bins: ', combo_bins)
print('Example of combo labels: ', combo_labels[113])

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

training_feature_file = 'Data/Data_for_ML/raw_features/test_parameters_1000v1.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(6))
training_features = np.vectorize(round_sigfigs)(training_features)
combo_labels = np.round(combo_labels, decimals=3)
print('Example of rounded combo labels: ', combo_labels[113])
print('Example of features: ', training_features[113])
# Shuffle the data properly
training_features, combo_labels = shuffle(training_features, combo_labels)

# Save the arrays aas a text file
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
testing_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/testing_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
np.savetxt(training_path + 'label_sub12_dndz_S', combo_labels, fmt='%.2f')
#np.savetxt(testing_path + 'label_sub12_dndz_S', combo_labels, fmt='%.2f')
np.savetxt(training_path + 'feature', training_features, fmt='%.2f')
#np.savetxt(testing_path + 'feature', testing_features, fmt='%.2f')
np.savetxt(bin_path + 'bin_sub12_dndz', combo_bins)

# plt.plot(kbins, training_kband[0], 'rx')
# kbins = kbins[0::2]
# training_kband = training_kband[0][0::2]
# plt.plot(kbins, training_kband, 'gx')
# plt.show()