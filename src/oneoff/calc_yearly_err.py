import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt


df = pd.read_feather("../../results/final_all_obs.feather")

years = range(1980,2021)
rmse_per_year = np.empty((len(years)))
obs_per_year = np.empty((len(years)))
lakes_per_year = np.empty((len(years)))
rmse_per_year[:] = np.nan
obs_per_year[:] = np.nan
lakes_per_year[:] = np.nan
def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean()) 
for ct,year in enumerate(years):
	print("year ",ct)
	year = str(year)
	year_df = df[df['date'].str.contains(year)]
	unique_lake = np.unique(year_df['site_id'].values)
	rmse_per_lake = [calc_rmse(year_df[year_df['site_id']==i_d]['pred'], year_df[year_df['site_id']==i_d]['actual']) for i_d in unique_lake]
	rmse_per_year[ct] = np.median(np.array(rmse_per_lake))
	lakes_per_year[ct] = len(unique_lake)
	obs_per_year[ct] = year_df.shape[0]

plt.plot(years,rmse_per_year)
plt.title("Median per-lake RMSE per year")
plt.xlabel("year")
plt.ylabel("RMSE (deg C)")
plt.show()
plt.clf()

plt.plot(years,obs_per_year)
plt.title("obs per year")
plt.xlabel("year")
plt.ylabel("number of obs")
plt.show()
plt.clf()


plt.plot(years,lakes_per_year)
plt.title("lakes observed per year")
plt.xlabel("year")
plt.ylabel("number of lakes")
plt.show()
plt.clf()



