from __future__ import print_function
from colossus.lss import bias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)


z = [0.692, 0.896, 1.144, 1.604, 1.836, 2.002]
iz_mapping = {194: 0.692, 182: 0.896, 169: 1.144, 152: 1.604, 146: 1.836, 142: 2.002}

# Import the GALFORM data
MCMCbias_path = 'Data/Data_for_ML/MCMC/halobias_100/'
MCMCbias_filenames = os.listdir(MCMCbias_path)
MCMCbias_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

for filename in MCMCbias_filenames:
    bias_path = os.path.join(MCMCbias_path, filename)
    mhalo_df = pd.read_csv(bias_path, delimiter=",")
    # Use the map function to create the new column
    mhalo_df['z'] = mhalo_df['iz'].map(iz_mapping)
    mhalo_df['bias'] = bias.haloBias(mhalo_df['mhhalo'], model='tinker10', z=mhalo_df['z'], mdef='vir')
    avgbias_df = mhalo_df.groupby('z')['bias'].mean().reset_index()
    plt.plot(avgbias_df['z'], avgbias_df['bias'], '-', c='tab:blue', alpha=0.3)
plt.xlim([0.69, 2.004])
plt.ylabel('<bias>')
plt.xlabel('Redshift, z')
plt.tight_layout()
plt.show()

min50 = np.load("Data/Data_for_ML/MCMC/min50idx.npy")
list_bias = []

merwispx = [0.7493765586034913, 2.3990024937655865]
merwispy = [1.230622168738361, 2.392363728920593]
merhizx = [0.750116940387823, 2.400608735205026]
merhizy = [1.2435400516795865, 2.4386304909560725]
for i in min50:
    bias_path = os.path.join(MCMCbias_path, f'training_galfrun.{i+1}.halobias')
    mhalo_df = pd.read_csv(bias_path, delimiter=",")
    # Use the map function to create the new column
    mhalo_df['z'] = mhalo_df['iz'].map(iz_mapping)
    mhalo_df['bias'] = bias.haloBias(mhalo_df['mhhalo'], model='tinker10', z=mhalo_df['z'], mdef='vir')
    avgbias_df = mhalo_df.groupby('z')['bias'].mean().reset_index()
    plt.plot(avgbias_df['z'], avgbias_df['bias'], '-', c='tab:blue', alpha=0.3)
    list_bias.append(avgbias_df['bias'][4])

    if i == 61:
        plt.plot(avgbias_df['z'], avgbias_df['bias'], '-', c='tab:red', zorder=10,
                 lw=3, label="Best GALFORM")
        print("Best model average bias values: ")
        print(avgbias_df)
plt.axline((merwispx[0], merwispy[0]), (merwispx[1], merwispy[1]), linestyle='dashed', c='tab:gray',
           label='Merson et al. 2019 WISP calibrated')
plt.axline((merhizx[0], merhizy[0]), (merhizx[1], merhizy[1]), linestyle='dotted', c='tab:gray',
           label='Merson et al. 2019 HiZELS calibrated')
plt.legend()
plt.xlim([0.69, 2.004])
plt.ylabel(r'<bias$_{eff}$>')
plt.xlabel('Redshift, z')
plt.tight_layout()
plt.savefig("Plots/50bestMCMC_bias.pdf")
plt.show()

print("\nRange of z=2.002 bias: ", min(list_bias), "-", max(list_bias))

# Find the percentile ranges
print("\n10th Percentile of bias: ", np.percentile(list_bias, 10))
print("90th Percentile of bias: ", np.percentile(list_bias, 90))
