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
	pdb.set_trace()
	tmp = 0
# site_df[(site_df['Date'].str.contains('-12-')) | (site_df['Date'].str.contains('-01-')) | (site_df['Date'].str.contains('-02-'))]
 #year since 1980 * 4 + season (winter1,spring2,summ3,fall4)

obs['bin_id'] = [createBinID(date) for date in obs['Date'].values]
 # for obs in obs