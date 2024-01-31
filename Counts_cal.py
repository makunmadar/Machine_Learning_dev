import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from Loading_functions import counts_generation
from scipy.interpolate import interp1d
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)


# Bagley data
names = ["Flux", "n", "+", "-"]
CHa_0916 = pd.read_csv("Data/Data_for_ML/Observational/Bagley_20/Corrected_Ha_Bagley_0916.csv",
                       delimiter=",", names=names, skiprows=1)
CHa_0916 = CHa_0916.astype(float)
CHa_0916 = np.log10(CHa_0916)
CHa_0916ytop = CHa_0916["+"] - CHa_0916["n"]
CHa_0916ybot = CHa_0916["n"] - CHa_0916["-"]

# Calculate the Euclid flux limit
f_e = 2  # x10^-16 erg^-1 s^-1 cm^-2

# Import the GALFORM data
MCMCcounts_path = 'Data/Data_for_ML/MCMC/counts_100_0918/counts_HaNII_ext/'
MCMCcounts_filenames = os.listdir(MCMCcounts_path)
MCMCcounts_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

list_counts, counts_bins = counts_generation(MCMCcounts_filenames, MCMCcounts_path)

for i in range(len(list_counts)):
    C_i = [j for j, e in enumerate(list_counts[i]) if e != 0]
    plt.plot(counts_bins[C_i], list_counts[i][C_i], '-', c='tab:blue', alpha=0.3)

plt.errorbar(CHa_0916["Flux"], CHa_0916["n"], yerr=(CHa_0916ybot, CHa_0916ytop),
             label=r"Bagley'20 Corrected H$\alpha+[NII]$", markeredgecolor='black',
             ecolor='black', capsize=2, fmt='o', zorder=11, markerfacecolor="None")
plt.ylabel("Log$_{10}$(N(>S) [deg$^{-2}$])")
plt.xlabel("Log$_{10}$(Flux [10$^{-16}$erg$^{-1}$  s$^{-1}$ cm$^{-2}$])")
plt.axvline(x=np.log10(f_e), linestyle='dashed', c='tab:gray', alpha=0.7, label='Euclid flux limit')
plt.xlim([0.2, 1.0])
plt.ylim([1, 4.2])
plt.tight_layout()
plt.savefig("Plots/100bestMCMC_counts.pdf")
#plt.text(0.5, 0.5, '0.692$\leq$z$\less$2.002', fontsize=21)
plt.show()

min50 = np.load("Data/Data_for_ML/MCMC/min50idx.npy")

for i in min50:

    C_i = [j for j, e in enumerate(list_counts[i]) if e != 0]
    plt.plot(counts_bins[C_i], list_counts[i][C_i], '-', c='tab:blue', alpha=0.3)

    if i == 61:
        plt.plot(counts_bins[C_i], list_counts[i][C_i], '-', c='tab:red',
                 zorder=10, lw=3, label="Best GALFORM")

plt.errorbar(CHa_0916["Flux"], CHa_0916["n"], yerr=(CHa_0916ybot, CHa_0916ytop),
             label=r"Bagley et al. 20 Corrected H$\alpha$+[NII]", markeredgecolor='black',
             ecolor='black', capsize=2, fmt='o', zorder=11, markerfacecolor="None")
plt.ylabel("Log$_{10}$(N(>S) [deg$^{-2}$])")
plt.xlabel("Log$_{10}$(Flux [10$^{-16}$erg$^{-1}$  s$^{-1}$ cm$^{-2}$])")
plt.axvline(x=np.log10(f_e), linestyle='dashed', c='tab:gray', alpha=0.7, label='Euclid flux limit')
plt.legend()
plt.xlim([0.2, 1.0])
plt.ylim([1, 4.2])
plt.tight_layout()
plt.text(0.8, 4, '0.9$\less$z$\less$1.8', c='tab:blue')
plt.text(0.8, 3.7, '0.9$\less$z$\less$1.6', c='black')
plt.savefig("Plots/50bestMCMC_counts.pdf")
plt.show()

# We will use interpolation to find the number counts corresponding to f > 2x10^-16 erg-1s-1cm-2
C_i = [j for j, e in enumerate(list_counts[61]) if e != 0]
interp_func = interp1d(counts_bins[C_i], list_counts[61][C_i], kind='linear', fill_value='extrapolate')
# Specify the x value target
targed_x = np.log10(2)
# Find corresponding y value
target_y = interp_func(targed_x)
print("Number count at Euclid limit for best model: ", 10**target_y, " deg^-2")

list_fcounts = []
for i in min50:
    C_i = [j for j, e in enumerate(list_counts[i]) if e != 0]
    interp_func = interp1d(counts_bins[C_i], list_counts[i][C_i], kind='linear', fill_value='extrapolate')
    target_y = interp_func(targed_x)
    list_fcounts.append(10**target_y)
    if 10**target_y == 1.0:
        print("Model: ", i)
print("\nRange of number counts: ", min(list_fcounts), "-", max(list_fcounts))

# Find the percentile ranges
print("\n10th Percentile of number counts: ", np.percentile(list_fcounts, 10))
print("90th Percentile of number counts: ", np.percentile(list_fcounts, 90))

# Save the 50 best parameter sets
df_100 = pd.read_csv('minMAE_100MCMC_LUMLIM.csv')
df_100 = df_100[::30]
df_50best = df_100.iloc[min50, 1:12]
df_50best.to_csv('50bestMCMCparameters.csv', sep=',', index=False)
