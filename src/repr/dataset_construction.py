import pandas as pd
import numpy as np
import sys
import os
import pdb


#load lakes
site_ids = np.load("../../metadata/lakeset.npy",allow_pickle=True)

#load obs
obs = pd.read_csv("../../data/raw/obs/lake_surface_temp_obs.csv")

#trim obs to sites
obs = obs[np.isin(obs['site_id'],site_ids)]
# 
#create train/test split
def createBinID(date):
	years_since = int(date[:4]) - 1980
	if '-12-' in date or '-01-' in date or '-02-' in date:
		return years_since*4+1
	elif '-03-' in date or '-04-' in date or '-05-' in date:
		return years_since*4+2
	elif '-06-' in date or '-07-' in date or '-08-' in date:
		return years_since*4+3
	elif '-09-' in date or '-10-' in date or '-11-' in date:
		return years_since*4+4
	else:
		print("err")
		pdb.set_trace()
		return None
# site_df[(site_df['Date'].str.contains('-12-')) | (site_df['Date'].str.contains('-01-')) | (site_df['Date'].str.contains('-02-'))]
 #year since 1980 * 4 + season (winter1,spring2,summ3,fall4)

obs['bin_id'] = [createBinID(date) for date in obs['Date'].values]
obs['subset'] = 'none'


for site_id in site_ids:
	site_df = obs[obs['site_id']==site_id]
	unq_bins,ct_per_unq_bin = np.unique(site_df['bin_id'],return_counts=True)

	#keep track of what we already added
	excluded_date_list = site_df['Date'].values
	included_date_list = []

	bin_ind = unq_bins.shape[0]-1
	obs_ct = 0
	while obs_ct < 100:
		bin_df = site_df[site_df['bin_ind'] == bin_ind]
		if ct_per_unq_bins[bin_ind] > 0
			pdb.set_trace()
	pdb.set_trace()
 # for obs in obs