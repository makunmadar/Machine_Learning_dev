import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)

min50 = np.load("Data/Data_for_ML/MCMC/min50idx.npy")

#  Lacey parameters
X_test = np.array([1.0, 320, 320, 3.4, 0.8, 0.74, 0.9, 0.3, 0.05, 0.005, 0.1])

best100 = pd.read_csv("minMAE_100MCMC_LUMLIM.csv")
best100 = best100.drop_duplicates(subset=['alpha_reheat'], ignore_index=True)

best50 = best100.iloc[min50]
drop_columns = ['MAE', 'redshift', 'subvolume', 'modelno', 'lum_lim']
best50 = best50.drop(drop_columns, axis=1)

dist_data = []

for i in range(len(best50)):
    euclidean_dist = euclidean(X_test, best50.iloc[i].values)
    cosine_dist = cosine_similarity([X_test], [best50.iloc[i].values])[0][0]
    MAE = np.sum(np.abs(best50.iloc[i].values - X_test)/X_test)
    dist_data.append([min50[i], euclidean_dist, cosine_dist, MAE])


columns=['index', 'euclid_dist', 'cosine_sim', 'MAE']
dist_df = pd.DataFrame(dist_data, columns=columns)
print(dist_df[dist_df['euclid_dist'] == dist_df['euclid_dist'].min()])
print(dist_df[dist_df['cosine_sim'] == dist_df['cosine_sim'].max()])
print(dist_df[dist_df['MAE'] == dist_df['MAE'].min()])
print("Best parameters:")
print(best50.loc[[35]])

#####
# # Work out the average halo mass
# WFIRST_halo = pd.read_csv('Data/Data_for_ML/MCMC/halobias_100_WFIRST/training_galfrun.62.halobias')
# print(WFIRST_halo['mhhalo'].mean())
# print(WFIRST_halo.groupby('iz').mean())
# print("\n")
#
# Euclid_halo = pd.read_csv('Data/Data_for_ML/MCMC/halobias_100/training_galfrun.62.halobias')
# print(Euclid_halo['mhhalo'].mean())
# print(Euclid_halo.groupby('iz').mean())