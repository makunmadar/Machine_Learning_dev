import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# Import luminosity bins
full_lum_file = 'multi_gal.lf.line/L_line_bins'
sub5_lum_file = 'multi_gal.lf.line/L_line_bins_subsamp5'
sub6_lum_file = 'multi_gal.lf.line/L_line_bins_subsamp6'

full_lum = genfromtxt(full_lum_file)
sub5_lum = genfromtxt(sub5_lum_file)
sub6_lum = genfromtxt(sub6_lum_file)

# Import LF values
full_LF_file = 'multi_gal.lf.line/Ha(dust)_LF'
sub5_LF_file = 'multi_gal.lf.line/Ha(dust)_LF_subsamp5'
sub6_LF_file = 'multi_gal.lf.line/Ha(dust)_LF_subsamp6'

full_LF = genfromtxt(full_LF_file)
sub5_LF = genfromtxt(sub5_LF_file)
sub6_LF = genfromtxt(sub6_LF_file)

# Plot the first lines LF
X_full = np.log10(full_lum)
full_LF[full_LF == 0] = 10**(-4.37)
Y_full = np.log10(full_LF) + np.log10(np.log(10))

X_sub5 = np.log10(sub5_lum)
Y_sub5 = np.log10(sub5_LF) + np.log10(np.log(10))
X_sub6 = np.log10(sub6_lum)
Y_sub6 = np.log10(sub6_LF) + np.log10(np.log(10))

fig, axs = plt.subplots(2,2, figsize=(10,10),
                        facecolor = 'w', edgecolor = 'k',
                        sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)
axs = axs.ravel()

for i in range(2):
    axs[i].plot(X_full, Y_full[i], 'r.', label = f'Full model {i+1}')
    axs[i+2].plot(X_full, Y_full[i], 'r.', label = f'Full model {i+1}')
    axs[i].plot(X_sub5, Y_sub5[i], 'gx--', alpha = 0.6, label = '5 subsamples')
    axs[i+2].plot(X_sub6, Y_sub6[i], 'bx--', alpha = 0.6, label = '6 subsamples')
    axs[i].set_ylim([-5.0, 0.0])
    axs[i+2].set_ylim([-5.0, 0.0])
    axs[i].legend()
    axs[i+2].legend()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel(r"Log$_{10}$(L$_{H\alpha}$) [10$^{40}$ h$^{-2}$ erg/s] ", fontsize = 15)
plt.ylabel(r"Log$_{10}$(dn/dln(L$_{H\alpha}$)) [h$^3$ Mpc$^{-3}$]", fontsize = 15)
plt.show()
#fig.savefig('Subsampled_LF_compare.pdf')


