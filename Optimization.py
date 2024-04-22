'''
Main MCMC script for calibration to observables
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from Loading_functions import lf_df, load_all_models
from numpy import genfromtxt
from scipy.interpolate import interp1d
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)


def mcmc_updater(curr_state, curr_likeli, curr_err, model, obs_x, obs_y, pred_bins, mae_weighting,
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
        obs_y: Observable y vector
        mae_weighting:
        sigma:
        curr_state (float): the current parameter/state value
        curr_likeli (float): the current likelihood estimate
        curr_err:
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
    # print("Proposed state: ", proposal_state)

    # Calculate the acceptance criterion
    # We want to minimize the MAE value but the likelihood
    # is set to be maximized therefore flip the acceptance criterion ratio
    prop_likeli, prop_pred, prop_err = likelihood(proposal_state, model, obs_x, obs_y, pred_bins, mae_weighting)

    # accept_crit = curr_err / prop_err
    # print("Prop: ", prop_likeli)
    # print("Curr: ", curr_likeli)
    accept_crit = prop_likeli / curr_likeli
    # print("Ratio: ", accept_crit)
    # print("\n")
    # Generate a random number between 0 and 1
    accept_threshold = np.random.uniform(0, 1)

    # If the acceptance criterion is greater than the random number,
    # accept the proposal state as the current state

    if accept_crit > accept_threshold:
        count = 1
        return proposal_state, prop_likeli, prop_pred, prop_err, count

    else:
        count = 0
        return curr_state, curr_likeli, curr_pred, curr_err, count


def metropolis_hastings(
        model, obs_x, obs_y, pred_bins, mae_weighting, likelihood,
        proposal_distribution, initial_state, num_samples, stepsize, burnin):
    """ Compute the Markov Chain Monte Carlo

    Args:
        model: Emulator model
        obs_y: Observable y values
        mae_weighting:
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the
          next proposal state
        initial_state (list): The initial conditions to start the chain
        num_samples (integer): The number of samples to compute,
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
    likeli = []
    errors = []
    counts = []

    # The number of samples in the burn in phase
    idx_burnin = int(burnin * num_samples)

    # Set the current state to the initial state
    curr_state = initial_state
    curr_likeli, curr_pred, curr_err = likelihood(curr_state, model, obs_x, obs_y,
                                                  pred_bins, mae_weighting)
    # predictions.append(curr_pred)
    # likeli.append(curr_likeli)

    for i in range(num_samples):
        # The proposal distribution sampling and comparison
        # occur within the mcmc_updater routine
        curr_state, curr_likeli, curr_pred, curr_err, count = mcmc_updater(
            curr_state=curr_state,
            curr_likeli=curr_likeli,
            curr_err=curr_err,
            model=model,
            obs_x=obs_x,
            obs_y=obs_y,
            pred_bins=pred_bins,
            # scaling=scaling,
            mae_weighting=mae_weighting,
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
            likeli.append(curr_likeli)
            errors.append(curr_err)
            counts.append(count)

    return samples, predictions, likeli, errors, counts


def likelihood(params, models, obs_x, obs_y, pred_bins, mae_weighting):
    """ Compute the MAE likelihood function, comparing a prediction to observed values

    Args:
        params: Current state parameters. Inputs to the emulator model.
        models: Multiple tensorflow models used to map the input Galform parameters to an output predicting,
        y-axis values
        obs_y: Observable y-axis vector
        mae_weighting: Observable weightings for MAE calculation

    Returns:
        weighted_mae: The calculated MAE value comparing the prediction plot to the observed plot
        predictions: for plotting

    """
    # Mean average error (MAE) loss
    # Load in the array of models and average over the predictions

    predictions_mean = np.mean([model(params) for model in models], axis=0)

    # Perform model prediction using the input parameters
    predictions = predictions_mean[0]

    # Calculate the mean absolute error (MAE)
    # First need to interpolate between the observed and the predicted
    # Create common x-axis with same axis as the observable x bins

    # Interpolate or resample the redshift distribution data onto the common x-axis
    interp_funcz = interp1d(pred_bins[0:49], predictions[0:49], kind='linear', fill_value='extrapolate')
    interp_yz = interp_funcz(obs_x[0:7])
    # interp_y1 = interp_funcz(obs_x)
    #
    # # Interpolate or resample the K-band luminosity function data onto the common x-axis
    interp_funck = interp1d(pred_bins[49:74], predictions[49:74], kind='linear', fill_value='extrapolate')
    interp_yk = interp_funck(obs_x[7:25])
    # interp_y1 = interp_funck(obs_x)
    #
    # # Interpolate or resample the K-band luminosity function data onto the common x-axis
    interp_funcr = interp1d(pred_bins[74:102], predictions[74:102], kind='linear', fill_value='extrapolate')
    interp_yr = interp_funcr(obs_x[25:45])
    # interp_y1 = interp_funck(obs_x)

    # Combine the interpolated y values
    pred = np.hstack([interp_yz, interp_yk, interp_yr])

    # Manually calculate the weighted MAE
    abs_diff = np.abs(obs_y - pred)  # / sigma # L1 norm
    # sqr_diff = ((pred-obs_y)**2)/sigma**2 # L2 norm
    weighted_diff = mae_weighting * abs_diff

    bag = weighted_diff[0:7] / 7
    driv_k = weighted_diff[7:25] / 18
    driv_r = weighted_diff[25:45] / 20

    weighted_err = (np.sum(bag) + np.sum(driv_k) + np.sum(driv_r)) * (1/3)

    # Convert to Laplacian likelihood
    b = 0.005
    likelihood = (1/(2*b)) * np.exp(-weighted_err/b)
    # likelihood = np.exp(-weighted_err/2)

    return likelihood, predictions_mean[0], weighted_err


def proposal_distribution(x, stepsize):
    # Select the proposed state (new guess) from a Laplacian distribution
    # centered at the current state, using scale parameter equal to 1/20th the parameter range
    min_bound = np.array([0.2, 10, 10, 1.0, 0.0, 0.1, 0.5, 0.2, 0.01, 0.001, 0.01])
    max_bound = np.array([3.0, 800, 800, 4.0, 4.0, 4.0, 1.2, 0.5, 0.3, 0.05, 0.2])

    # Making sure the proposed values are within the minmax boundary
    while True:

        proposal_params = x + np.random.laplace(scale=stepsize)

        if np.all((proposal_params >= min_bound) & (proposal_params <= max_bound)):
            break

    return proposal_params


# Load in the Observational data
bag_headers = ["z", "n", "+", "-"]
Ha_b = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Ha_Bagley_dndz.csv",
                   delimiter=",", names=bag_headers, skiprows=1)
Ha_b = Ha_b.astype(float)
# sigmaz = (Ha_b["+"].values - Ha_b['-'].values)/2
#
# # Try on the Driver et al. 2012 LF data
driv_headers = ['Mag', 'LF', 'error', 'Freq']
drive_pathk = 'Data/Data_for_ML/Observational/Driver_12/lfk_z0_driver12.data'
df_k = lf_df(drive_pathk, driv_headers, mag_low=-23.75, mag_high=-15.25)
df_k = df_k[(df_k != 0).all(1)]
df_k['LF'] = df_k['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_k['error'] = df_k['error'] * 2  # Same reason
# sigmak = df_k['error'].values
#
drive_pathr = 'Data/Data_for_ML/Observational/Driver_12/lfr_z0_driver12.data'
df_r = lf_df(drive_pathr, driv_headers, mag_low=-23.25, mag_high=-13.75)
df_r = df_r[(df_r != 0).all(1)]
df_r['LF'] = df_r['LF'] * 2  # Driver plotted in 0.5 magnitude bins so need to convert it to 1 mag.
df_r['error'] = df_r['error'] * 2  # Same reason
# sigmar = df_r['error'].values

# Import Cole et al. 2001
# cole_headers = ['Mag', 'PhiJ', 'errorJ', 'PhiK', 'errorK']
# cole_path_k = 'Data/Data_for_ML/Observational/Cole_01/lfJK_Cole2001.data'
# df_ck = lf_df(cole_path_k, cole_headers, mag_low=-24.00-1.87, mag_high=-16.00-1.87)
# df_ck = df_ck[df_ck['PhiK'] != 0]
# df_ck = df_ck.sort_values(['Mag'], ascending=[True])
# df_ck['Mag'] = df_ck['Mag'] + 1.87

# Combine the observational data
obs_x = np.hstack([Ha_b['z'].values, df_k['Mag'].values, df_r['Mag'].values])
# obs_x = df_k['Mag'].values
# obs_x = Ha_b['z'].values
obs_y = np.log10(np.hstack([Ha_b['n'].values, df_k['LF'].values, df_r['LF'].values]))
# obs_y = df_k['LF'].values
# obs_y = Ha_b['n']
# sigma = np.hstack([sigmaz, sigmak, sigmar])
# obs_y = np.load("Scaled_obs.npy")
# sigma = np.load("fractional_sigma.npy")
# scaling = np.load("Scaling_factor.npy")
# offset = np.load("Offset.npy")

# obs_y = np.load("Lacey_y_true.npy")

# # MAE weighting
# W = [8.0] * 49 + [1.0] * 16
W = [6.0] * 7 + [1.0] * 18 + [1.0] * 20
param_range = [2.8, 790.0, 790.0, 3.0, 4.0, 3.0, 0.7, 0.3, 0.29, 0.049, 0.19]
b_step = [i/20 for i in param_range]

# Load the Galform bins
bin_file = 'Data/Data_for_ML/bin_data/bin_full'
bins = genfromtxt(bin_file)

# Load in the neural network
members = load_all_models(n_models=5)
print('Loaded %d models' % len(members))

# np.random.seed(42)

num_samples = int(15000)
burnin = 0.5
n_walkers = 10

n_samples = []
n_predictions = []
n_likelihoods = []
n_error = []

start = time.perf_counter()
# Wrap this in multiple walkers:
for n in range(n_walkers):
    # Initial state is the random starting point of the input parameters.
    # The prior is tophat uniform between the parameter bounds so initial
    # states are uniformly chosen at random between these bounds.
    initial_ar = np.random.uniform(0.2, 3.0)
    initial_vd = np.random.uniform(10, 800)
    initial_vb = np.random.uniform(10, 800)
    initial_ah = np.random.uniform(1.0, 4.0)
    initial_ac = np.random.uniform(0.0, 4.0)
    initial_ns = np.random.uniform(0.1, 4.0)
    initial_Fs = np.random.uniform(0.5, 1.2)
    initial_fe = np.random.uniform(0.2, 0.5)
    initial_fb = np.random.uniform(0.01, 0.3)
    initial_fS = np.random.uniform(0.001, 0.05)
    initial_tb = np.random.uniform(0.01, 0.2)
    initial_state = np.array([initial_ar, initial_vd, initial_vb, initial_ah, initial_ac, initial_ns,
                              initial_Fs, initial_fe, initial_fb, initial_fS, initial_tb])
    initial_state = initial_state.reshape(1, -1)

    # Generate samples over the posterior distribution using the metropolis_hastings function
    samples, predictions, likelihoods, error, counts = metropolis_hastings(
        model=members,
        obs_x=obs_x,
        obs_y=obs_y,
        pred_bins=bins,
        # scaling=scaling,
        mae_weighting=W,
        likelihood=likelihood,
        proposal_distribution=proposal_distribution,
        initial_state=initial_state,
        num_samples=num_samples,
        stepsize=b_step,
        burnin=burnin
    )
    n_samples.append(samples)
    n_predictions.append(predictions)
    n_likelihoods.append(likelihoods)
    n_error.append(error)
    accept_percentage = (np.sum(counts))*100/len(counts)
    print("Percentage of accepted proposals: ", accept_percentage, "%")

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds' % elapsed, ' for MCMC')

flattened_predictions = np.reshape(n_predictions, (-1, 102))

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0)
axs = axs.ravel()

for n in range(n_walkers):
    samples = np.array(n_samples[n])
    first_dim_size = samples.shape[0]
    samples_reshape = samples.reshape(first_dim_size, 11)

    alpha_reheat = [i[0] for i in samples_reshape]
    vhotdisk = [i[1] for i in samples_reshape]
    vhotburst = [i[2] for i in samples_reshape]
    alpha_hot = [i[3] for i in samples_reshape]
    alpha_cool = [i[4] for i in samples_reshape]
    nu_sf = [i[5] for i in samples_reshape]
    axs[0].plot(range(first_dim_size), alpha_reheat, label=f'Walker: {n+1}')
    # axs[0].axhline(y=1.0, c="red")
    axs[0].set_ylabel(r"$\alpha_{ret}$")
    axs[0].set_ylim(0.2, 3.0)
    # axs[1].axhline(y=320, c="red")
    axs[1].plot(range(first_dim_size), vhotdisk)
    axs[1].set_ylabel(r"$V_{SN, disk}$")
    axs[1].set_ylim(10, 800)
    axs[2].plot(range(first_dim_size), vhotburst)
    # axs[2].axhline(y=320, c="red")
    axs[2].set_ylabel(r"$V_{SN, burst}$")
    axs[2].set_ylim(10, 800)
    axs[3].plot(range(first_dim_size), alpha_hot)
    # axs[3].axhline(y=3.4, c="red")
    axs[3].set_ylabel(r"$\gamma_{SN}$")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylim(1.0, 4.0)
    axs[4].plot(range(first_dim_size), alpha_cool)
    # axs[4].axhline(y=0.8, c="red")
    axs[4].set_ylabel(r"$\alpha_{cool}$")
    axs[4].set_xlabel("Iteration")
    axs[4].set_ylim(0.0, 4.0)
    axs[5].plot(range(first_dim_size), nu_sf)
    # axs[5].axhline(y=0.74, c="red")
    axs[5].set_ylabel(r"$\nu_{SF}$ [Gyr$^{-1}$]")
    axs[5].set_xlabel("Iteration")
    axs[5].set_ylim(0.1, 4.0)

fig.legend(loc=7)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10),
                        facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0)
axs = axs.ravel()

for n in range(n_walkers):
    samples = np.array(n_samples[n])
    first_dim_size = samples.shape[0]
    samples_reshape = samples.reshape(first_dim_size, 11)

    Fstab = [i[6] for i in samples_reshape]
    fellip = [i[7] for i in samples_reshape]
    fburst = [i[8] for i in samples_reshape]
    fSMBH = [i[9] for i in samples_reshape]
    tau_burst = [i[10] for i in samples_reshape]
    axs[0].plot(range(first_dim_size), Fstab, label=f'Walker: {n+1}')
    # axs[0].axhline(y=1.0, c="red")
    axs[0].set_ylabel(r"Fstab")
    axs[0].set_ylim(0.5, 1.2)
    # axs[1].axhline(y=320, c="red")
    axs[1].plot(range(first_dim_size), fellip)
    axs[1].set_ylabel(r"fellip")
    axs[1].set_ylim(0.2, 0.5)
    axs[2].plot(range(first_dim_size), fburst)
    # axs[2].axhline(y=320, c="red")
    axs[2].set_ylabel(r"fburst$")
    axs[2].set_ylim(0.01, 0.3)
    axs[3].plot(range(first_dim_size), fSMBH)
    # axs[3].axhline(y=3.4, c="red")
    axs[3].set_ylabel(r"fSMBH")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylim(0.001, 0.05)
    axs[4].plot(range(first_dim_size), tau_burst)
    # axs[4].axhline(y=0.8, c="red")
    axs[4].set_ylabel(r"tau_burst")
    axs[4].set_xlabel("Iteration")
    axs[4].set_ylim(0.01, 0.2)

fig.legend(loc=7)
plt.show()

# Flatten the samples
flattened_samples = np.reshape(n_samples, (-1, 11))
flattened_likelihoods = np.reshape(n_likelihoods, (-1, 1))
flattened_error = np.reshape(n_error, (-1, 1))

# np.save('Samples_redshiftdist.npy', flattened_samples)
# np.save('Likelihoods_redshiftdist.npy', flattened_likelihoods)
# np.save('Predictions_redshiftdist.npy', flattened_predictions)
# np.save('Samples_KLF.npy', flattened_samples)
# np.save('Likelihoods_KLF.npy', flattened_likelihoods)
# np.save('Predictions_KLF.npy', flattened_predictions)

np.save('Samples_combo_MAE611_5test.npy', flattened_samples)
np.save('Likelihoods_combo_MAE611_5test.npy', flattened_likelihoods)
np.save('Predictions_combo_MAE611_5test.npy', flattened_predictions)
np.save('Predictions_combo_MAE611_5test_raw.npy', n_predictions)
np.save('Error_combo_MAE611_5test.npy', flattened_error)
np.save('Error_combo_611_5test_raw.npy', n_error)  # For the future

# np.save('Error_comboz_MAE.npy', flattened_error)
# np.save('Error_combok_MAE.npy', flattened_error)
