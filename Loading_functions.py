"""
Useful custom functions
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d
import re


def dndz_df(path, columns):
    """
    This function extracts the redshift distribution GALFORM data and
    saves the relevant flux limited data it in a dataframe.

    :param path: path to the distribution file
    :param columns: names of the redshift distribution columns
    :return: dataframe
    """

    # Extract only the data that has a flux limit of 2E7 Jy (or as close as possible)
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

    # Create a pandas dataframe of the extract n(z) data
    data = np.vstack(data)
    df = pd.DataFrame(data=data, columns=columns)
    df = df.apply(pd.to_numeric)

    # Filter according to our redshift requirements
    # Don't want to include potential zero values at z=0.692
    # df = df[(df['z'] < 2.1)]
    df = df[(df['z'] > 0.7)]

    # Transform into log_10 form (ignoring 0's to not create NANs)
    df['dN(>S)/dz'] = np.log10(df['dN(>S)/dz'].mask(df['dN(>S)/dz'] <= 0)).fillna(0)

    return df


def lf_df(path, columns, mag_low, mag_high):
    """
    This function extracts just the LF data from GALFORM output
    gal.lf files and saves it in a dataframe.

    :param path: path to the LF file
    :param columns: what are the names of the magnitude columns?
    :return: dataframe
    """

    # Extract the data within the file
    data = []
    with open(path, 'r') as fh:
        for curline in fh:
            if curline.startswith("#"):
                header = curline
            else:
                row = curline.strip().split()
                data.append(row)

    # Convert extracted data into a pandas dataframe
    data = np.vstack(data)
    df = pd.DataFrame(data=data)
    df = df.apply(pd.to_numeric)
    df.columns = columns

    # Only account for the data within the magnitude ranges given
    df = df[(df['Mag'] <= mag_high)]
    df = df[(df['Mag'] >= mag_low)]
    return df


def masked_mae(y_true, y_pred):
    """
    Custom MAE loss function for emulator during training.
    The main function of the mask is ignoring any missing datapoints
    particularly within the LF data.
    The MAE is manually calculated per sample

    Args:
        y_true: True Galform data across the three statistics
        y_pred: Predicted Galform statistics

    Returns: Tensorflow loss value

    """
    # # Create a mask where non-zero values are True
    mask = tf.not_equal(y_true, -100)
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    # Calculate the MAE loss
    loss = tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))

    # # Split the data into n(z), LFk and LFr
    # y_truez = y_true[:7]
    # y_truek = y_true[7:25]
    # y_truer = y_true[25:]
    # y_predz = y_pred[:7]
    # y_predk = y_pred[7:25]
    # y_predr = y_pred[25:]
    #
    # # Create a mask where non-zero values of n(z) are True
    # maskz = tf.not_equal(y_truez, 0)
    # y_truez = tf.boolean_mask(y_truez, maskz)
    # y_predz = tf.boolean_mask(y_predz, maskz)
    # # Scale y_truez to [0, 1] using Min-Max scaling
    # y_truez_min = tf.keras.backend.min(y_truez)
    # y_truez_max = tf.keras.backend.max(y_truez)
    # y_truez_scaled = (y_truez - y_truez_min) / (y_truez_max - y_truez_min)
    # y_predz_scaled = (y_predz - y_truez_min) / (y_truez_max - y_truez_min)
    # z_loss = tf.reduce_mean(tf.abs(y_truez_scaled - y_predz_scaled))
    #
    # # Create a mask where non-zero values of LFk are True
    # maskk = tf.not_equal(y_truek, 0)
    # y_truek = tf.boolean_mask(y_truek, maskk)
    # y_predk = tf.boolean_mask(y_predk, maskk)
    # # Scale y_truek to [0, 1] using Min-Max scaling
    # y_truek_min = tf.keras.backend.min(y_truek)
    # y_truek_max = tf.keras.backend.max(y_truek)
    # y_truek_scaled = (y_truek - y_truek_min) / (y_truek_max - y_truek_min)
    # y_predk_scaled = (y_predk - y_truek_min) / (y_truek_max - y_truek_min)
    # k_loss = tf.reduce_mean(tf.abs(y_truek_scaled - y_predk_scaled))
    #
    # # Create a mask where non-zero values of LFr are True
    # maskr = tf.not_equal(y_truer, 0)
    # y_truer = tf.boolean_mask(y_truer, maskr)
    # y_predr = tf.boolean_mask(y_predr, maskr)
    # # Scale y_truer to [0, 1] using Min-Max scaling
    # y_truer_min = tf.keras.backend.min(y_truer)
    # y_truer_max = tf.keras.backend.max(y_truer)
    # y_truer_scaled = (y_truer - y_truer_min) / (y_truer_max - y_truer_min)
    # y_predr_scaled = (y_predr - y_truer_min) / (y_truer_max - y_truer_min)
    # r_loss = tf.reduce_mean(tf.abs(y_truer_scaled - y_predr_scaled))
    #
    # y_true_scaled = tf.concat([y_truez_scaled, y_truek_scaled, y_truer_scaled], axis=0)
    # y_pred_scaled = tf.concat([y_predz_scaled, y_predk_scaled, y_predr_scaled], axis=0)
    #
    # loss = tf.reduce_mean(tf.abs(y_true_scaled - y_pred_scaled))
    return loss


def load_all_models(n_models):
    """
    Load all the emulator models from file and combine into a list of models

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_9x5_upmask_2899_LRELU_int'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'masked_mae': masked_mae},
                                           compile=False)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models


def predict_all_models(n_models, X_test, variant):
    """
    Load all the models from file and create predictions.

    Args:
        n_models: number of models in the ensemble
        X_test: Test sample in np.array already. No need to normalize as this is done in the model
        variant: string of model name

    Returns:
        all_yhat: list of predictions from each model
    """

    all_yhat = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + variant
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={"masked_mae": masked_mae}, compile=False)
        print('>loaded %s' % filename)
        # Produce prediction
        yhat = model(X_test)
        all_yhat.append(yhat)

    return all_yhat


def find_number(text, c):
    '''
    Identify the model number of a path string

    :param text: model name as string
    :param c: after what string symbol does the number reside
    :return: the model number
    '''

    return [int(s) for s in re.findall(r'%s(\d+)' % c, text)]


def dndz_generation(galform_filenames, galform_filepath, O_df, column_headers):
    """
    Generating the redhisft distribution data for emulator training.
    Reading the GALFORM output files, and interpolating so the bins
    are equal to the observables (Bagley et al. 2020) bins.
    Output the associated model numbers for sanity check.

    Args:
        galform_filenames: List of GALFORM output redshift distribution file names
        galform_filepath: Path to GALFORM output n(z) data files
        O_df: Observable dataframe
        column_headers: Column names used for dndz_df function

    Returns: List of generated n(z) values, list of model numbers

    """

    # Define empty array for storing training n(z) into
    list_dndz = np.empty((0, 7))
    model_list = []

    # Go through each n(z) file in list
    for file in galform_filenames:
        model_number = find_number(file, '.')
        model_list.append(model_number[0])

        # Extract the relevant n(z) data with custom function
        df_z = dndz_df(galform_filepath + file, column_headers)

        # Use the interpolation package to transform the data into observable bin frame
        interp_funcz = interp1d(df_z['z'].values, df_z['dN(>S)/dz'].values, kind='linear', fill_value='extrapolate')
        interp_yz1 = interp_funcz(O_df['z'].values)
        interp_yz1[O_df['z'].values > max(df_z['z'].values)] = 0
        interp_yz1[O_df['z'].values < min(df_z['z'].values)] = 0

        list_dndz = np.vstack([list_dndz, interp_yz1])
        # list_dndz = np.vstack(([list_dndz, df_z['dN(>S)/dz']]))
        # dndz_bins = df_z['z'].values

    return list_dndz, model_list  # , dndz_bins


def LF_generation(galform_filenames, galform_filepath, O_dfk, O_dfr, column_headers):
    """
    Generating the luminosity function data for emulator training.
    Reading the Galform output files and interpolating so the bins
    are equal to the observables (Driver et al. 2012) bins.
    This is repeated for the K-band and r-band.

    Args:
        galform_filenames: List of GALFORM gal.lf filenames
        galform_filepath: Path to GALFORM output gal.lf data files
        O_dfk: Driver et al. 2012 dataframe for K-band
        O_dfr: Driver et al. 2012 dataframe for r-band
        column_headers:

    Returns: List of K-band and r-band values concatenated

    """

    # Define empty array for storing training LF data
    list_lf = np.empty((0, 38))

    # Go through each gal.lf file in list
    for file in galform_filenames:
        model_number = find_number(file, '.')

        # Extract LF data using custom function
        df_lfk = lf_df(galform_filepath + file, column_headers, mag_low=-23.80,
                       mag_high=-15.04)  # -23.80, -15.04 for Driver
        # Process the K-band values
        df_lfk['Krdust'] = np.log10(df_lfk['Krdust'].mask(df_lfk['Krdust'] <= 0)).fillna(0)
        df_lfk = df_lfk[df_lfk['Krdust'] != 0]

        # Interpolate into the observable reference frame
        interp_funck = interp1d(df_lfk['Mag'].values, df_lfk['Krdust'].values, kind='linear', fill_value='extrapolate',
                                bounds_error=False)
        interp_yk1 = interp_funck(O_dfk['Mag'].values)
        interp_yk1[O_dfk['Mag'].values < min(df_lfk['Mag'].values)] = 0

        df_lfr = lf_df(galform_filepath + file, column_headers, mag_low=-23.36,
                       mag_high=-13.73)  # -23.36, -13.73 for Driver
        df_lfr['Rrdust'] = np.log10(df_lfr['Rrdust'].mask(df_lfr['Rrdust'] <= 0)).fillna(0)
        df_lfr = df_lfr[df_lfr['Rrdust'] != 0]
        interp_funcr = interp1d(df_lfr['Mag'].values, df_lfr['Rrdust'].values, kind='linear', fill_value='extrapolate',
                                bounds_error=False)
        interp_yr1 = interp_funcr(O_dfr['Mag'].values)
        interp_yr1[O_dfr['Mag'].values < min(df_lfk['Mag'].values)] = 0

        # Calculate the amount of padding needed for both arrays
        # paddingk = [(0, 18 - len(interp_yk1))]  # 18 for driver
        # paddingr = [(0, 20 - len(interp_yr1))]

        # Pad both arrays with zeros
        # padded_arrayk = np.pad(interp_yk1, paddingk, mode='constant', constant_values=0)
        # padded_arrayr = np.pad(interp_yr1, paddingr, mode='constant', constant_values=0)

        # Combine the K-band and r-band training LF values
        lf_vector = np.concatenate((interp_yk1, interp_yr1))
        # lf_vector = np.concatenate((df_lfk['Krdust'], df_lfr['Rrdust']))
        list_lf = np.vstack([list_lf, lf_vector])

        # lf_bins = np.concatenate((df_lfk['Mag'].values, df_lfr['Mag'].values))

    return list_lf  # , lf_bins
