import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from joblib import load
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import corner
import time


def masked_mae(y_true, y_pred):
    # The tensorflow models custom metric, this won't affect the predictions
    # But it gets rid of the warning message
    mask = tf.not_equal(y_true, 0)  # Create a mask where non-zero values are True
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))

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
        filename = 'Models/Ensemble_model_' + str(i + 1) + '_2512_mask'
        # Load model from file
        model = tf.keras.models.load_model(filename, custom_objects={'masked_mae': masked_mae},
                                           compile=False)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models


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

    df = df[(df['Mag'] <= -18.00)]
    df = df[(df['Mag'] >= -25.11)]
    return df


def mcmc_updater(curr_state, curr_likeli, model, obs_x, obs_y, pred_bins,
                 likelihood, proposal_distribution, stepsize, curr_pred):
    """ Propose a new state and compare the likelihoods

    Given the current state (initially random),
      current likelihood, the likelihood function, and
      the transition (proposal) distribution, `mcmc_updater` generates
      a new proposal, evaluate its likelihood, compares that to the current
      likelihood with a uniformly samples threshold,
    then it returns new or current state in the MCMC chain.

    Args:
        model: Tensorflow model
        obs_x: Observable x vector
        obs_y: Observable y vector
        pred_bins: Fixed prediction x bins
        curr_state (float): the current parameter/state value
        curr_likeli (float): the current likelihood estimate
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the
          next proposal state
        curr_pred: current predictions

    Returns:
        (tuple): either the current state or the new state
          and its corresponding likelihood and prediction
    """
    # Generate a proposal state using the proposal distribution
    # Proposal state == new guess state to be compared to current
    proposal_state = proposal_distribution(curr_state, stepsize)

    # Calculate the acceptance criterion
    # We want to minimize the MAE value but the likelihood
    # is set to be maximized therefore flip the acceptance criterion ratio
    prop_likeli, prop_pred = likelihood(proposal_state, model, obs_x, obs_y, pred_bins)
    accept_crit = curr_likeli / prop_likeli

    # Generate a random number between 0 and 1
    accept_threshold = np.random.uniform(0, 1)

    # If the acceptance criterion is greater than the random number,
    # accept the proposal state as the current state

    if accept_crit > accept_threshold:
        return proposal_state, prop_likeli, prop_pred

    # Else
    return curr_state, curr_likeli, curr_pred


def metropolis_hastings(
        model, obs_x, obs_y, pred_bins, likelihood, proposal_distribution, initial_state,
        num_samples, stepsize, burnin):
    """ Compute the Markov Chain Monte Carlo

    Args:
        model: Emulator model
        obs_x: Observable x values
        obs_y: Observable y values
        pred_bins: Prediction x-axis bins
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the
          next proposal state
        initial_state (list): The initial conditions to start the chain
        num_samples (integer): The number of samples to compte,
          or length of the chain
        burnin (float): a float value from 0 to 1.
          The percentage of chain considered to be the burnin length

    Returns:
        samples (list): The Markov Chain,
          samples from the posterior distribution
        preds (list): Predictions for plotting
    """
    samples = []
    predictions = []

    # The number of samples in the burn in phase
    idx_burnin = int(burnin * num_samples)

    # Set the current state to the initial state
    curr_state = initial_state
    curr_likeli, curr_pred = likelihood(curr_state, model, obs_x, obs_y, pred_bins)
    predictions.append(curr_pred)

    for i in range(num_samples):
        # The proposal distribution sampling and comparison
        # occur within the mcmc_updater routine
        curr_state, curr_likeli, curr_pred = mcmc_updater(
            curr_state=curr_state,
            curr_likeli=curr_likeli,
            model=model,
            obs_x=obs_x,
            obs_y=obs_y,
            pred_bins=pred_bins,
            likelihood=likelihood,
            proposal_distribution=proposal_distribution,
            stepsize=stepsize,
            curr_pred=curr_pred
        )

        # Append the current state to the list of samples
        if i >= idx_burnin:
            # Only append after the burn in to avoid including
            # parts of the chain that are prior_dominated
            samples.append(curr_state)
            predictions.append(curr_pred)

    return samples, predictions


def likelihood(params, model, obs_x, obs_y, pred_bins):
    """ Compute the MAE likelihood function, comparing a prediction to observed values

    Args:
        params: Current state parameters. Inputs to the emulator model.
        model: A tensorflow model used to map the input Galform parameters to an output predicting,
        y-axis values
        obs_x: Observable x-axis vector
        obs_y: Observable y-axis vector
        pred_bins: Fixed x-axis bins corresponding to the predicted y-axis values

    Returns:
        weighted_mae: The calculated MAE value comparing the prediction plot to the observed plot
        predictions: for plotting

    """
    # Mean average error (MAE) loss
    # Load in the array of models and average over the predictions
    ensemble_pred = list()
    for model in members:
        # Perform model prediction using the input parameters
        pred = model.predict(params)
        ensemble_pred.append(pred)
    predictions = np.mean(ensemble_pred, axis=0)

    # Perform model prediction using the input parameters
    # predictions = model.predict(params)
    predictions = predictions[0]

    # Calculate the mean absolute error (MAE)
    # First need to interpolate between the observed and the predicted
    # Create common x-axis with same axis as the observable x bins

    # Interpolate or resample the redshift distribution data onto the common x-axis
    interp_funcz = interp1d(pred_bins[0:13], predictions[0:13], kind='linear', fill_value='extrapolate')
    interp_yz1 = interp_funcz(obs_x[0:7])

    # Interpolate or resample the luminosity function data onto the common x-axis
    interp_funck = interp1d(pred_bins[13:22], predictions[13:22], kind='linear', fill_value='extrapolate')
    interp_yk1 = interp_funck(obs_x[7:19])

    # Combine the interpolated y values
    interp_y1 = np.hstack([interp_yz1, interp_yk1])

    # Working out the MAE values
    # In the future I want to update this to work with the errors from the observables
    # error_weightsz = 1/sigma
    weighted_mae = mean_absolute_error(obs_y, interp_y1)

    return weighted_mae, predictions


def proposal_distribution(x, stepsize):
    # Select the proposed state (new guess) from a Gaussian distribution
    # centered at the current state, within a Gaussian of width 'stepsize'

    # Making sure the proposed values are within the minmax boundary
    while True:

        proposal_params = np.random.normal(x, stepsize)

        if np.all((proposal_params >= 0) & (proposal_params <= 1)):
            break

    return proposal_params


# Load in the Observational data
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

# Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_path = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = kband_df(drive_path, driv_headers)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
df_k['error_upper'] = np.log10(df_k['LF'] + df_k['error']) - np.log10(df_k['LF'])
df_k['error_lower'] = np.log10(df_k['LF']) - np.log10(df_k['LF'] - df_k['error'])
df_k['LF'] = np.log10(df_k['LF'])

# Combine the observational data
obs_x = np.hstack([Ha_b['z'].values, df_k['Mag'].values])
obs_y = np.hstack([Ha_b['n'].values, df_k['LF'].values])

# Load in the minmax scaler for the parameter data
scaler_feat = load("mm_scaler_feat.bin")

# Load the Galform bins
bin_file = 'Data/Data_for_ML/bin_data/bin_sub12_dndz'
bins = genfromtxt(bin_file)

# Load in the neural network
# For now only loading one of the models. Sort out the averaging later
# Checking the model is loading in correctly
# model = tf.keras.models.load_model('Models/Ensemble_model_1_2512_mask',
#                                    custom_objects={"masked_mae": masked_mae}, compile=False)
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))

np.random.seed(42)

num_samples = int(4000)
stepsize = 0.005
burnin = 0.0  # For now testing with zero burn in
n_walkers = 2

n_samples = []

start = time.perf_counter()
# Wrap this in multiple walkers:
for n in range(n_walkers):
    # Initial state is the random starting point of the input parameters.
    # The prior is tophat uniform between the parameter bounds so initial
    # states are uniformly chosen at random between these bounds.
    initial_state = np.random.uniform(0, 1, size=6)
    initial_state = initial_state.reshape(1, -1)
    # Generate samples over the posterior distribution using the metropolis_hastings function
    samples, predictions = metropolis_hastings(
        model=members,
        obs_x=obs_x,
        obs_y=obs_y,
        pred_bins=bins,
        likelihood=likelihood,
        proposal_distribution=proposal_distribution,
        initial_state=initial_state,
        num_samples=num_samples,
        stepsize=stepsize,
        burnin=burnin
    )
    n_samples.append(samples)

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds' % elapsed, ' for MCMC ')

# colour = iter(cm.rainbow(np.linspace(0, 1, len(predictions))))
# # This is for a large number of samples we want to plot,
# # easier to see the trend by subsampling
# # for theta in samples[np.random.randint(len(samples), size=100)]:
# for theta in predictions:
#     c = next(colour)
#     plt.plot(bins[0:13], theta[0:13], c=c, alpha=0.1)
# plt.errorbar(Ha_b["z"], Ha_b["n"], yerr=(Ha_ybot, Ha_ytop), markeredgecolor='black', ecolor='black', capsize=2,
#              fmt='co', label="Bagley'20 Observed")
# plt.xlabel('Redshift')
# plt.ylabel('Log$_{10}$(dN(>S)/dz) [deg$^{-2}$]')
# plt.xlim(0.7, 2.0)
# plt.legend()
# plt.show()
#
# colour = iter(cm.rainbow(np.linspace(0, 1, len(predictions))))
# for theta in predictions:
#     c = next(colour)
#     plt.plot(bins[13:22], theta[13:22], color=c, alpha=0.1)
# plt.errorbar(df_k['Mag'], df_k['LF'], yerr=(df_k['error_lower'], df_k['error_upper']),
#              markeredgecolor='black', ecolor='black', capsize=2, fmt='co', label='Driver et al. 2012')
# plt.xlabel(r"M$_{AB}$ - 5log(h)", fontsize=16)
# plt.ylabel(r"Log$_{10}$(LF (Mpc/h)$^{-3}$ (mag$_{AB}$)$^{-1}$)", fontsize=16)
# plt.xlim(-18, -25)
# plt.ylim(-6, -1)
# plt.legend()
# plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0)
axs = axs.ravel()

for n in range(n_walkers):
    samples = np.array(n_samples[n])
    first_dim_size = samples.shape[0]
    samples_reshape = samples.reshape(first_dim_size, 6)
    samples_reshape = scaler_feat.inverse_transform(samples_reshape)

    alpha_reheat = [i[0] for i in samples_reshape]
    vhotdisk = [i[1] for i in samples_reshape]
    vhotburst = [i[2] for i in samples_reshape]
    alpha_hot = [i[3] for i in samples_reshape]
    alpha_cool = [i[4] for i in samples_reshape]
    nu_sf = [i[5] for i in samples_reshape]
    axs[0].plot(range(first_dim_size), alpha_reheat)
    axs[1].plot(range(first_dim_size), vhotdisk)
    axs[2].plot(range(first_dim_size), vhotburst)
    axs[3].plot(range(first_dim_size), alpha_hot)
    axs[4].plot(range(first_dim_size), alpha_cool)
    axs[5].plot(range(first_dim_size), nu_sf)

plt.show()

# labels = ['alpha_reheat', 'Vhotdisk', 'Vhotburst', 'alpha_hot', 'alpha_cool', 'nu_sf']
# For multiple walkers the shape of "samples" should have the shape (num_walkers, num_samples, num_parameters)

# Flatten the samples
# flattened_samples = np.reshape(n_samples, (-1, 6))

# Create the corner plot
# fig = corner.corner(samples_reshape, show_titles=True, labels=labels,
#                     plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
# fig.savefig("corner.png")
