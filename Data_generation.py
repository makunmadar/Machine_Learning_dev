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


def counts_df(path, columns):
    '''
    This function extracts just the counts data and saves it in a dataframe.

    :param path: path to the counts file
    :param columns: what are the names of the emission line columns?
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


# # Number counts
# base_path_counts = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_counts/"
#
# # The following columns are for the number counts
# columns_N = ['S_nu', 'dN/dln(S_nu)', 'N(>S)']
#
# base_filenames = os.listdir(base_path_counts)
# base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
#
# empty_Hacounts = np.empty((0, 6))
# for file in base_filenames:
#     model_number = find_number(file, '.')  # This is in the form of a string
#     df = counts_df(base_path_counts + file, columns_N)
#
#     # Convert the flux from Jy to erg s-1 cm-2 for S_nu
#     df['S_nu'] = df["S_nu"] * 1e-23 * 1e16  # Flux[10^-16 erg s-1 cm-2]
#     df['S_nu'] = np.log10(df['S_nu'].replace(0, 1e-20))
#     df['N(>S)'] = np.log10(df['N(>S)'].replace(0, 1e-20))
#
#     # Using previous work and Bagley 2020 data to find the counts range
#     lower = min(df['S_nu'], key=lambda x: abs(x - 0))
#     upper = min(df['S_nu'], key=lambda x: abs(x - 1))
#
#     df = df[df['S_nu'].between(lower, upper)]
#     idx = np.round(np.linspace(0, len(df) - 1, 6)).astype(int)
#     counts_vector = df['N(>S)'].values
#     counts_vector = counts_vector[idx]
#     empty_Hacounts = np.vstack([empty_Hacounts, counts_vector])
#
# countbins = df['S_nu'].values
# countbins = countbins[idx]
# print('Number count bins: ', countbins)
# print('Example of number count values: ', empty_Hacounts[0])
# print('\n')

# Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_training/"

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

# Combine the two data sets with the parameter data
#combo_bins = np.hstack([countbins, dndzbins]) # This data is not required for the machine learning
#combo_Ha = np.hstack([empty_Hacounts, empty_Hadndz])

# testing_feature_file1 = 'Data/Data_for_ML/raw_features/test_parameters.csv'
# testing_features1 = genfromtxt(testing_feature_file1, delimiter=',', skip_header=1)
# # Import the second feature file not including the redshift, subvolume or model information
# testing_feature_file2 = 'Data/Data_for_ML/raw_features/test_parameters_extended_v3.csv'
# testing_features2 = genfromtxt(testing_feature_file2, delimiter=',', skip_header=1, usecols= range(6))
# # Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
# #print(testing_features2)
# testing_features2 = testing_features2[::30]
# #print(testing_features2[::30])
# testing_features = np.vstack([testing_features1, testing_features2])

training_feature_file = 'Data/Data_for_ML/raw_features/test_parameters_1000v1.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(6))
# As we don't have the full range of training data yet:
training_features = training_features[:400, :]
training_features = np.vectorize(round_sigfigs)(training_features)
training_Hadndz = np.round(training_Hadndz, decimals=2)

# Shuffle the data properly
training_features, training_Hadndz = shuffle(training_features, training_Hadndz)

# Save the arrays aas a text file
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
testing_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/testing_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
np.savetxt(training_path + 'label_sub12_dndz', training_Hadndz, fmt='%.2f') # Only saving the redshift distribution so far
#np.savetxt(testing_path + 'label_sub12_dndz', training_Hadndz, fmt='%.2f')
np.savetxt(training_path + 'feature', training_features, fmt='%.2f')
#np.savetxt(testing_path + 'feature', testing_features, fmt='%.2f')
np.savetxt(bin_path + 'bin_sub12_dndz', dndzbins)

# for i in range(len(training_Hadndz)):
#
#     #plt.plot(dndzbins, y_test[i])
#     plt.plot(dndzbins, training_Hadndz[i])
#
# plt.title("Spread of training redshift distribution data")
# plt.xlabel("Redshift, z", fontsize=15)
# plt.ylabel("Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]", fontsize=15)
# plt.show()


