import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt


# df = pd.read_feather("../../results/final_all_obs.feather")
df = pd.read_feather("../../results/all_outputs_and_obs_031821.feather")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")
# obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")
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
# for ct,year in enumerate(years):
# 	print("year ",ct)
# 	year = str(year)
# 	year_df = df[df['Date'].str.contains(year)]
# 	unique_lake = np.unique(year_df['site_id'].values)
# 	rmse_per_lake = [calc_rmse(year_df[year_df['site_id']==i_d]['wtemp_predicted-ealstm'], year_df[year_df['site_id']==i_d]['wtemp_actual']) for i_d in unique_lake]
# 	rmse_per_year[ct] = np.median(np.array(rmse_per_lake))
# 	lakes_per_year[ct] = len(unique_lake)
# 	obs_per_year[ct] = year_df.shape[0]


au_rmse = []
n_au = 0
sp_rmse = []
n_sp = 0
su_rmse = []
n_su = 0
wi_rmse = []
n_wi = 0


plot_df = pd.DataFrame()
plot_df['season'] = ['winter','spring','summer','fall']


for ct,site_id in enumerate(site_ids):
	print(ct)
	site_df = df[df['site_id'] == site_id]
	site_df_wi = site_df[(site_df['Date'].str.contains('-12-')) | (site_df['Date'].str.contains('-01-')) | (site_df['Date'].str.contains('-02-'))]
	if site_df_wi.shape[0] > 0:
		n_wi += site_df_wi.shape[0]
		wi_rmse.append(calc_rmse(site_df_wi['wtemp_predicted-ealstm'].values, site_df_wi['wtemp_actual']))
	site_df_sp = site_df[(site_df['Date'].str.contains('-03-')) | (site_df['Date'].str.contains('-04-')) | (site_df['Date'].str.contains('-05-'))]
	if site_df_sp.shape[0] > 0:
		n_sp += site_df_sp.shape[0]
		sp_rmse.append(calc_rmse(site_df_sp['wtemp_predicted-ealstm'].values, site_df_sp['wtemp_actual']))
	site_df_su = site_df[(site_df['Date'].str.contains('-06-')) | (site_df['Date'].str.contains('-07-')) | (site_df['Date'].str.contains('-08-'))]
	if site_df_su.shape[0] > 0:
		n_su += site_df_su.shape[0]
		su_rmse.append(calc_rmse(site_df_su['wtemp_predicted-ealstm'].values, site_df_su['wtemp_actual']))
	site_df_au = site_df[(site_df['Date'].str.contains('-09-')) | (site_df['Date'].str.contains('-10-')) | (site_df['Date'].str.contains('-11-'))]
	if site_df_au.shape[0] > 0:
		n_au += site_df_au.shape[0]
		au_rmse.append(calc_rmse(site_df_au['wtemp_predicted-ealstm'].values, site_df_au['wtemp_actual']))
pdb.set_trace()

wi_rmse = np.array(wi_rmse)
sp_rmse = np.array(sp_rmse)
su_rmse = np.array(su_rmse)
au_rmse = np.array(au_rmse)
medians = [np.median(wi_rmse), np.median(sp_rmse),np.median(su_rmse),np.median(au_rmse)]
print("counts")
print(n_wi)
print(n_sp)
print(n_su)
print(n_au)
print("medians")
print(repr(medians))
plot_df['median rmse'] = medians
#Meteorological spring includes March, April, and May;
# meteorological summer includes June, July, and August;
# meteorological fall includes September, October, and November; and
# meteorological winter includes December, January, and February.

plt.bar(plot_df['season'], plot_df['median rmse'], color='blue')
plt.xlabel("season")
plt.ylabel("Median RMSE")
plt.title("RMSE per season")
plt.show()
# plt.plot(years,rmse_per_year)
# plt.title("Median per-lake RMSE per year")
# plt.xlabel("year")
# plt.ylabel("RMSE (deg C)")
# plt.show()
# plt.clf()

# plt.plot(years,obs_per_year)
# plt.title("obs per year")
# plt.xlabel("year")
# plt.ylabel("number of obs")
# plt.show()
# plt.clf()


plt.plot(years,lakes_per_year)
plt.title("lakes observed per year")
plt.xlabel("year")
plt.ylabel("number of lakes")
plt.show()
plt.clf()



