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


for site_ct,site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids))
    site_df = obs[obs['site_id']==site_id]
    if site_df.shape[0] < 100:
        print("not enough")
        pdb.set_trace()
    unq_bins,ct_per_unq_bin = np.unique(site_df['bin_id'],return_counts=True)

    #keep track of what we already added
    excluded_date_list = site_df['Date'].values
    included_date_list = []

    bin_ind = unq_bins.shape[0]-1
    obs_ct = 0
    while obs_ct < 100:
        if obs_ct == 50:
            pdb.set_trace()
        bin_df = site_df[site_df['bin_id'] == unq_bins[bin_ind]]
        if ct_per_unq_bin[bin_ind] > 0:
            #trim bin df to not select twice
            bin_df = bin_df[np.logical_not(np.isin(bin_df['Date'],included_date_list))]

            #select random
            included_date_list.append(bin_df.iloc[np.random.choice(np.arange(bin_df.shape[0]),1)]['Date'].values[0])
            
            #update counts
            ct_per_unq_bin[bin_ind] -= 1
            obs_ct += 1
        if bin_ind == 0:
            bin_ind = unq_bins.shape[0]-1
        else:
            bin_ind -= 1

    site_df_new = site_df[np.isin(site_df['Date'],included_date_list)]
    site_df_new = site_df_new.sort_values(by='Date')
    site_df_new['subset'] = ['train' if i >= 30 else 'test' for i in range(100)]
    site_df_new = site_df_new.reset_index()
    site_df_new.to_feather("../../data/raw/obs/"+site_id+"_100obs.feather")


 # for obs in obs