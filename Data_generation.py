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
from Loading_functions import dndz_generation, LF_generation, lf_df


def round_sigfigs(x):
    """
    Round a ndarray to 3 significant figures
    :param x: ndarray
    :return: rounded ndarray
    """
    return np.around(x, -int(np.floor(np.log10(abs(x)))) + 2)


#################
training_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/training_data/"
bin_path = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/bin_data/"
#################

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
frac_sigma = sigma / obs

# Import Cole et al. 2001
# cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
# cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
# df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00-1.87, mag_high=-16.00-1.87)
# df_ck = df_ck[df_ck['PhiK'] != 0]
# df_ck = df_ck.sort_values(['Mag'], ascending=[True])
# df_ck['Mag'] = df_ck['Mag'] + 1.87
# np.save('fractional_sigma.npy', frac_sigma)

##################################

# Redshift distribution
columns_Z = ["z", "d^2N/dln(S_nu)/dz", "dN(>S)/dz"]
base_path_dndz = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_dndz_HaNIIext_2999/dndz_HaNII_ext/"

base_filenames = os.listdir(base_path_dndz)
base_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_Hadndz, model_numbers = dndz_generation(galform_filenames=base_filenames, galform_filepath=base_path_dndz,
                                                 O_df=Ha_b, column_headers=columns_Z)
model_numbers = [x - 1 for x in model_numbers]

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

base_path_lf = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_kband_training/LF_2999/LF/"
basek_filenames = os.listdir(base_path_lf)
basek_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_lf = LF_generation(galform_filenames=basek_filenames, galform_filepath=base_path_lf,
                            O_dfk=df_k, O_dfr=df_r, column_headers=columns_lf)

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

training_feature_file = 'Data/Data_for_ML/raw_features/updated_parameters_extended_3000v4.csv'
training_features = genfromtxt(training_feature_file, delimiter=',', skip_header=1, usecols=range(11))
# Note that due to the extra columns there are duplicates of the parameters that need to be taken care of
training_features = training_features[::30]
#training_features = np.delete(training_features, 1470, axis=0)  # As for now we don't have model 1471
training_features = np.take(training_features, model_numbers, axis=0)  # As we don't have some models
print('Length of features: ', len(training_features))

# training_features = np.vectorize(round_sigfigs)(training_features)
# combo_labels = np.round(combo_labels, decimals=3)
# print('Example of rounded combo labels: ', combo_labels[113])

# Shuffle the data properly
# training_features, combo_labels = shuffle(training_features, combo_labels)

# Save the arrays as a text file
np.savetxt(training_path + 'label_full2999_int', combo_labels)
np.savetxt(training_path + 'feature_2999', training_features)
np.savetxt(bin_path + 'bin_full_int', combo_bins)

# Save individual physics data and bins for testing
# np.savetxt(training_path + 'label_dndz1999_int', training_Hadndz)
# np.savetxt(bin_path + 'bin_dndz_int', dndzbins)

