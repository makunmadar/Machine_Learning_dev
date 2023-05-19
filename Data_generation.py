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
import matplotlib.pyplot as plt


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
    df['Krdust'] = np.log10(df['Krdust'].replace(0, 1e-20))

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
    return np.around(x, -int(np.floor(np.log10(abs(x))))+2)


# Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_training_1000/"

base_filenames = os.listdir(base_path_dndz)
base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_Hadndz = np.empty((0, 12))
for file in base_filenames:
    model_number = find_number(file, '.')
    df = dndz_df(base_path_dndz + file, columns_Z)

    df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].replace(0, 1e-20))

    # Keep it between the redshift range 0.69<z<2.00
    lower = min(df['z'], key=lambda x: abs(x - 0.71))
    upper = min(df['z'], key=lambda x: abs(x - 2.00))

    df = df[df["z"].between(lower, upper)]
    idx = np.round(np.linspace(0, len(df) - 1, 12)).astype(int)
    dndz_vector = df['dN(>S)/dz'].values
    dndz_vector = dndz_vector[idx]
    training_Hadndz = np.vstack([training_Hadndz, dndz_vector])

dndzbins = df['z'].values
dndzbins = dndzbins[idx]
print('Redshift distribution bins: ', dndzbins)
print('Example of dn/dz values: ', training_Hadndz[0])

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

training_kband = np.empty((0, 12))

for file in basek_filenames:
    model_number = find_number(file, '.')
    df_k = kband_df(base_path_kband+file, columns_k)

    # Keep it between the magnitude range -18<Mag_k<-25
    lower_k = min(df_k['Mag'], key=lambda x: abs(x - (-25.0)))
    upper_k = min(df_k['Mag'], key=lambda x: abs(x - (-18.0)))

    df_k = df_k[df_k["Mag"].between(lower_k, upper_k)]
    idx = np.round(np.linspace(0, len(df_k) - 1, 12)).astype(int)
    k_vector = df_k['Krdust'].values
    k_vector = k_vector[idx]
    training_kband = np.vstack([training_kband, k_vector])

kbins = df_k['Mag'].values
kbins = kbins[idx]
print('k-band LF distribution bins: ', kbins)
print('Example of k-band LF values: ', training_kband[0])

# Combine the two data sets with the parameter data
combo_bins = np.hstack([dndzbins, kbins]) # This data is not required for the machine learning
combo_labels = np.hstack([training_Hadndz, training_kband])
print('Combo bins: ', combo_bins)
print('Example of combo labels: ', combo_labels[0])

# testing_feature_file1 = 'Data/Data_for_ML/raw_features/test_parameters.csv'
# testing_features1 = genfromtxt(testing_feature_file1, delimiter=',', skip_header=1)
# # Import the second feature file not including the redshift, subvolume or model information
# testing_feature_file2 = 'Data/Data_for_ML/raw_features/test_parameters_extended_v3.csv'
# testing_features2 = genfromtxt(testing_feature_file2, delimiter=',', skip_header=1, usecols= range(6))
# # Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
# testing_features2 = testing_features2[::30]
# testing_features = np.vstack([testing_features1, testing_features2])

training_feature_file = 'Data/Data_for_ML/raw_features/test_parameters_1000v1.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(6))
training_features = np.vectorize(round_sigfigs)(training_features)
combo_labels = np.round(combo_labels, decimals=2)

# Shuffle the data properly
training_features, combo_labels = shuffle(training_features, combo_labels)

# Save the arrays aas a text file
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
testing_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/testing_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
np.savetxt(training_path + 'label_sub12_dndz', combo_labels, fmt='%.2f') # Only saving the redshift distribution so far
#np.savetxt(testing_path + 'label_sub12_dndz', combo_labels, fmt='%.2f')
np.savetxt(training_path + 'feature', training_features, fmt='%.2f')
#np.savetxt(testing_path + 'feature', testing_features, fmt='%.2f')
#np.savetxt(bin_path + 'bin_sub12_dndz', combo_bins)

# for i in range(len(training_Hadndz)):
#
#     #plt.plot(dndzbins, y_test[i])
#     plt.plot(dndzbins, training_Hadndz[i])
#
# plt.title("Spread of training redshift distribution data")
# plt.xlabel("Redshift, z", fontsize=15)
# plt.ylabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
# plt.show()


