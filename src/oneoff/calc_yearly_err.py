import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt


# df = pd.read_feather("../../results/final_all_obs.feather")
# df = pd.read_feather("../../results/all_outputs_and_obs_031821.feather")
df = pd.read_feather("../../results/all_outputs_and_obs_041921.feather")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")
obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_040821.feather")

site_ids = metadata['site_id'].values #CHANGE DIS---------
df['Date'] = [str(d) for d in df['Date'].values]
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
	year_df = df[df['Date'].str.contains(year)]
	unique_lake = np.unique(year_df['site_id'].values)
	rmse_per_lake = [calc_rmse(year_df[year_df['site_id']==i_d]['wtemp_predicted-ealstm'], year_df[year_df['site_id']==i_d]['wtemp_actual']) for i_d in unique_lake]
	rmse_per_year[ct] = np.median(np.array(rmse_per_lake))
	lakes_per_year[ct] = len(unique_lake)
	obs_per_year[ct] = year_df.shape[0]


au_rmse = []
n_au = 0
sp_rmse = []
n_sp = 0
su_rmse = []
n_su = 0
wi_rmse = []
n_wi = []


# for site_id in 
#Meteorological spring includes March, April, and May;
# meteorological summer includes June, July, and August;
# meteorological fall includes September, October, and November; and
# meteorological winter includes December, January, and February.

# plt.bar(df['cluster'], df['Median RMSE'], color='blue')
# plt.xlabel("Cluster")
# plt.ylabel("Median RMSE")
# plt.title("RMSE per Cluster")
# plt.show()
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



