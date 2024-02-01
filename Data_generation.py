"""
Taking in the raw redshift distribution and LF outputs and formatting them appropriately for
machine learning training and evaluation.
"""
import numpy as np
import pandas as pd
import os
import re
from sklearn.utils import shuffle
from numpy import genfromtxt
from scipy.interpolate import interp1d
from Loading_functions import dndz_generation, LF_generation, lf_df


def round_sigfigs(x):
    """
    Round a ndarray to 3 significant figures
    :param x: ndarray
    :return: rounded ndarray
    """
    return np.around(x, -int(np.floor(np.log10(abs(x)))) + 2)


#################
# Define the path to save the training data
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
#################

# Load in the Observational data

# Import Bagley et al. 2020
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
# sigmaz = (Ha_b["+"].values - Ha_b['-'].values) / 2

# Import the Driver et al. 2012 data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path_k = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_path_k, driv_headers, mag_low=-23.75, mag_high=-15.25)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
# sigmak = df_k['error'].values

drive_path_r = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_path_r, driv_headers, mag_low=-23.25, mag_high=-13.75)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
# sigmar = df_r['error'].values

# sigma = np.hstack([sigmaz, sigmak, sigmar])
# obs = np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values])
# frac_sigma = sigma / obs

# Import Cole et al. 2001
# cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
# cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
# df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00-1.87, mag_high=-16.00-1.87)
# df_ck = df_ck[df_ck['PhiK'] != 0]
# df_ck = df_ck.sort_values(['Mag'], ascending=[True])
# df_ck['Mag'] = df_ck['Mag'] + 1.87
# np.save('fractional_sigma.npy', frac_sigma)

##################################

# H-alpha Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_HaNIIext_2999/dndz_HaNII_ext/"

base_filenames = os.listdir(base_path_dndz)
base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_Hadndz, model_numbers, dndzbins = dndz_generation(galform_filenames=base_filenames,
                                                           galform_filepath=base_path_dndz,
                                                           column_headers=columns_Z)
model_numbers = [x - 1 for x in model_numbers]

# dndzbins = Ha_b['z'].values
print('Redshift distribution bins: ', dndzbins)
print('Example of dn/dz values: ', training_Hadndz[1670])

# The following saves the min and max values from the entire n(z) dataset for scaling. Currently not used.
training_Hadndz = np.array(training_Hadndz)
# nzmin = np.min(training_Hadndz)
# nzmax = np.max(training_Hadndz)
# print("Min: ", nzmin)
# print("Max: ", nzmax)
#
# np.save('Data/Data_for_ML/min_nz_scale.npy', nzmin)
# np.save('Data/Data_for_ML/max_nz_scale.npy', nzmax)

# Scaling the n(z) data if required. This can be commented out.
# for i in range(len(training_Hadndz)):
#     training_Hadndz[i][training_Hadndz[i] == 0] = -100
#     non_zero_mask = training_Hadndz[i] != -100
#     # Manual scaling
#     training_Hadndz[i][non_zero_mask] = (training_Hadndz[i][non_zero_mask] - nzmin) / (nzmax - nzmin)
#
# print("Scaled labels", training_Hadndz[1671])

# K and r-band LF
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

base_path_lf = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_kband_training/LF_2999/LF/"
basek_filenames = os.listdir(base_path_lf)
basek_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_lf, lfbins= LF_generation(galform_filenames=basek_filenames, galform_filepath=base_path_lf,
                                   column_headers=columns_lf)

# lfbins = np.concatenate((df_k['Mag'].values, df_r['Mag'].values))
print('LF distribution bins: ', lfbins)
print('Example of k-band LF values: ', training_lf[1670][0:25])
print('Example of r-band LF values: ', training_lf[1670][25:53])

# The following saves the min and max from the K-band and r-band datasets. Not currently used.
# training_lfk = [i[0:18] for i in training_lf]
# training_lfr = [i[18:38] for i in training_lf]
# training_lfk = np.array(training_lfk)
# training_lfr = np.array(training_lfr)
# kmin = np.min(training_lfk)
# kmax = np.max(training_lfk)
# np.save('Data/Data_for_ML/min_k_scale.npy', kmin)
# np.save('Data/Data_for_ML/max_k_scale.npy', kmax)
# rmin = np.min(training_lfr)
# rmax = np.max(training_lfr)
# np.save('Data/Data_for_ML/min_r_scale.npy', rmin)
# np.save('Data/Data_for_ML/max_r_scale.npy', rmax)

# Applying the scaling to our data. This can be commented out if required.
# for i in range(len(training_lfk)):
#     training_lfk[i][training_lfk[i] == 0] = -100
#     non_zero_mask = training_lfk[i] != -100
#     training_lfk[i][non_zero_mask] = (training_lfk[i][non_zero_mask] - kmin) / (kmax - kmin)
#     training_lfr[i][training_lfr[i] == 0] = -100
#     non_zero_mask = training_lfr[i] != -100
#     training_lfr[i][non_zero_mask] = (training_lfr[i][non_zero_mask] - rmin) / (kmax - rmin)
# training_lf = np.hstack([training_lfk, training_lfr])
# print("Scaled labels", training_lf[1671])

# Combine the two data sets with the parameter data
combo_bins = np.hstack([dndzbins, lfbins])  # This data is not required for the machine learning
combo_labels = np.hstack([training_Hadndz, training_lf])

print('Combo bins: ', combo_bins)
print('Example of combo labels: ', combo_labels[1670])

# Load and process the input feature data from my parameters file used for generating the GALFORM runs
# These parameters were generated using a latin hypercube. 
training_feature_file = 'Data/Data_for_ML/raw_features/updated_parameters_extended_3000v4.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(11))
# Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
# The file contains columns for redshift snapshot, subvolume and model numbers so we will need to factor these out.
training_features = training_features[::30]

training_features = np.take(training_features, model_numbers, axis=0)  # As we don't have some models (1471)
print('Length of features: ', len(training_features))

# training_features = np.vectorize(round_sigfigs)(training_features)
# combo_labels = np.round(combo_labels, decimals=3)
# print('Example of rounded combo labels: ', combo_labels[113])

# Shuffle the data properly
# training_features, combo_labels = shuffle(training_features, combo_labels)


# Save the arrays as a text file
np.savetxt(training_path + 'label_full2999', combo_labels)
np.savetxt(training_path + 'feature_2999', training_features)
np.savetxt(bin_path + 'bin_full', combo_bins)

# Save individual physics data and bins for testing
# np.savetxt(training_path + 'label_dndz2999_int_scaled', training_Hadndz)
# np.savetxt(bin_path + 'bin_dndz_int', dndzbins)
