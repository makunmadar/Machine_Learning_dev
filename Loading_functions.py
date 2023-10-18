import numpy as np
import pandas as pd
import tensorflow as tf


def lf_df(path, columns, mag_low, mag_high):
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

    df = df[(df['Mag'] <= mag_high)]
    df = df[(df['Mag'] >= mag_low)]
    return df

def masked_mae(y_true, y_pred):
    mask = tf.not_equal(y_true, 0) # Create a mask where non-zero values are True
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))
    # frac_sig = np.load("fractional_sigma.npy")
    # frac_sig = tf.convert_to_tensor(frac_sig, dtype=tf.float32)

    # # Redshift distribution MAE:
    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])
    #
    # y_truez = tf.slice(y_true, begin=[0], size=[7])
    # y_predz = tf.slice(y_pred, begin=[0], size=[7])
    # # sig_z = tf.slice(frac_sig, begin=[0], size=[7])
    # maskz = tf.not_equal(y_truez, 0)  # Create a mask where non-zero values are True
    # masked_yz_true = tf.boolean_mask(y_truez, maskz)
    # masked_yz_pred = tf.boolean_mask(y_predz, maskz)
    # # masked_sig_z = tf.boolean_mask(sig_z, maskz)
    # min_zvalue = tf.reduce_min(masked_yz_true)
    # max_zvalue = tf.reduce_max(masked_yz_true)
    # scaling_factorz = 1.0 / (max_zvalue - min_zvalue)
    # offsetz = -min_zvalue * scaling_factorz
    # scaled_yz_pred = (masked_yz_pred * scaling_factorz) + offsetz
    # scaled_yz_true = (masked_yz_true * scaling_factorz) + offsetz
    #
    # # K-band LF MAE:
    # y_truek = tf.slice(y_true, begin=[7], size=[18])
    # y_predk = tf.slice(y_pred, begin=[7], size=[18])
    # # sig_k = tf.slice(frac_sig, begin=[7], size=[18])
    # maskk = tf.not_equal(y_truek, 0)  # Create a mask where non-zero values are True
    # masked_yk_true = tf.boolean_mask(y_truek, maskk)
    # masked_yk_pred = tf.boolean_mask(y_predk, maskk)
    # # masked_sig_k = tf.boolean_mask(sig_k, maskk)
    # min_kvalue = tf.reduce_min(masked_yk_true)
    # max_kvalue = tf.reduce_max(masked_yk_true)
    # scaling_factork = 1.0 / (max_kvalue - min_kvalue)
    # offsetk = -min_kvalue * scaling_factork
    # scaled_yk_pred = (masked_yk_pred * scaling_factork) + offsetk
    # scaled_yk_true = (masked_yk_true * scaling_factork) + offsetk
    #
    # # r-band LF MAE:
    # y_truer = tf.slice(y_true, begin=[25], size=[20])
    # y_predr = tf.slice(y_pred, begin=[25], size=[20])
    # # sig_r = tf.slice(frac_sig, begin=[25], size=[20])
    # maskr = tf.not_equal(y_truer, 0)  # Create a mask where non-zero values are True
    # masked_yr_true = tf.boolean_mask(y_truer, maskr)
    # masked_yr_pred = tf.boolean_mask(y_predr, maskr)
    # # masked_sig_r = tf.boolean_mask(sig_r, maskr)
    # min_rvalue = tf.reduce_min(masked_yr_true)
    # max_rvalue = tf.reduce_max(masked_yr_true)
    # scaling_factorr = 1.0 / (max_rvalue - min_rvalue)
    # offsetr = -min_rvalue * scaling_factorr
    # scaled_yr_pred = (masked_yr_pred * scaling_factorr) + offsetr
    # scaled_yr_true = (masked_yr_true * scaling_factorr) + offsetr
    #
    # scaled_y_pred = tf.concat([scaled_yz_pred, scaled_yk_pred, scaled_yr_pred], axis=-1)
    # scaled_y_true = tf.concat([scaled_yz_true, scaled_yk_true, scaled_yr_true], axis=-1)
    # # sigma = tf.concat([masked_sig_z, masked_sig_k, masked_sig_r], axis=-1)
    # abs_diff = tf.abs(scaled_y_pred - scaled_y_true)
    # loss = tf.reduce_mean(abs_diff)

    return loss


def load_all_models(n_models):
    """
    Load all the models from file

    :param n_models: number of models in the ensemble
    :return: list of ensemble models
    """

    all_models = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_55_mask_900_LRELU_int_up'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'masked_mae': masked_mae},
                                           compile=False)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models


def predict_all_models(n_models, X_test):
    """
    Load all the models from file and create predictions.

    Args:
        n_models: number of models in the ensemble
        X_test: Test sample in np.array already. No need to normalize as this is done in the model

    Returns:
        all_yhat: list of predictions from each model
    """

    all_yhat = list()
    for i in range(n_models):
        # Define filename for this ensemble
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_55_mask_900_LRELU_int_up'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={"masked_mae": masked_mae}, compile=False)
        print('>loaded %s' % filename)
        # Produce prediction
        yhat = model(X_test)
        all_yhat.append(yhat)

    return all_yhat
