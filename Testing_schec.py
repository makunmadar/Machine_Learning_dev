import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import re


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
    :return: Schecter funtion phi(L)
    """

    a = -1.3
    #b = 0.5
    phi_L = (np.log(10) / 2.5) * Ps * ((10 ** (0.4 * (Ms - M))) ** (a + 1)) * np.exp(-(10 ** (0.4 * (Ms - M))) ** b)
    return phi_L


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

    return df


def find_number(text, c):
    '''
    Identify the model number of a path string

    :param text: model name as string
    :param c: after what string symbol does the number reside
    :return: the model number
    '''

    return re.findall(r'%s(\d+)' % c, text)


# K-band LF column names
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

base_path_kband = "/home/dtsw71/PycharmProjects/ML/Data/Data_for_ML/raw_kband_testing/k_band_ext/"
basek_filenames = os.listdir(base_path_kband)
basek_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

training_kband = np.empty((0, 18))
training_kerror = np.empty((0, 18))

for file in basek_filenames:
    model_number = find_number(file, '.')
    df_k = kband_df(base_path_kband + file, columns_k)

    df_k = df_k[(df_k['Mag'] <= -17.67)]
    df_k = df_k[(df_k['Mag'] >= -25.11)]

    k_vector = df_k['Krdust'].values
    k_vector_errors = df_k['Krdust(error)'].values

    training_kband = np.vstack([training_kband, k_vector])
    training_kerror = np.vstack([training_kerror, k_vector_errors])

kbins = df_k['Mag'].values

# Plot the results
fig, axs = plt.subplots(3, 3, figsize=(18, 13),
                        facecolor='w', edgecolor='k')
fig.subplots_adjust(wspace=0.2, hspace=0.2)
axs = axs.ravel()

m = 150
for j in range(9):

    # Only want to test on one of the examples
    y = training_kband[j+m]
    err = training_kerror[j+m]
    zeroidx = [i for i, e in enumerate(y) if e == 0]
    print(zeroidx)
    x = kbins[y > 0]
    err = err[y > 0]
    y = y[y > 0]

    # Which bins to calibrate the curve fitting tool
    #idx = [0, (len(y)-1)//2, len(y)-1]
    x_sec = x[0:4]#[x[i] for i in idx]
    y_sec = y[0:4]#[y[i] for i in idx]
    err_sec = err[0:4]#[err[i] for i in idx]

    #params, cov = curve_fit(phi, x_sec, y_sec, bounds=((0, -22, 0), (10, -15, 1)), sigma=err_sec)
    params, cov = curve_fit(lin, x_sec, y_sec, sigma=err_sec)
    print("Predicted phi*: ", params[0])
    print("Predicted M*: ", params[1])
    #print("Predicted b: ", params[2])

    axs[j].scatter(x, y, color='skyblue', label='Data for model ' + str(j + 1 + m))
    axs[j].scatter(x_sec, y_sec, color='skyblue', edgecolors='black', label='Curve fit inputs')

    pred_bins = list()
    pred_phi = list()
    for i in zeroidx:
        pred_bins.append(kbins[i])
        #phi_p = phi(kbins[i], params[0], params[1], params[2])
        phi_p = lin(kbins[i], params[0], params[1])
        pred_phi.append(phi_p)

    axs[j].plot(pred_bins, pred_phi, 'ro', label='Fit')
    #axs[j].plot(kbins, phi(kbins, params[0], params[1]), 'r--', label="Full Schechter", alpha=0.4)
    axs[j].plot(kbins, lin(kbins, params[0], params[1]), 'r--', label="Full Schechter", alpha=0.4)
    axs[j].set_yscale('log')
    axs[j].legend()
    axs[j].set_ylim([None, 1])

plt.show()
