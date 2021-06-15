import pandas as pd 
import numpy as np
import os
import pdb

metadata = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
site_ids = metadata['site_id'].values

sites = []
rmses = []
final_df = pd.DataFrame()
def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean()) 
for site_ct,site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids), " site")
    if not metadata[metadata['site_id'] == site_id]['observed'].values[0]:
        print("unobserved")
        continue
    else:
        df = pd.read_feather("../../results/SWT_results_COLD_DEBUG/outputs_"+site_id+"_COLD_DEBUG.feather") 
        loss_outputs = df['temp_pred'].values[np.isfinite(df['temp_actual'].values)]
        loss_actual = df['temp_actual'].values[np.isfinite(df['temp_actual'].values)]
        loss_dates = df['index'].values[np.isfinite(df['temp_actual'].values)]
        sites.append(site_id)
        rmse = calc_rmse(loss_outputs, loss_actual)
        print("rmse: ",rmse)
        rmses.append(rmse)
    #    spring runs from March 1 to May 31;
        # summer runs from June 1 to August 31;
        # fall (autumn) runs from September 1 to November 30; and.
        # winter runs from December 1 to February 28 (February 29 in a leap year).

        df = pd.DataFrame(data={'site_id':site_id, 'date':loss_dates, 'pred':loss_outputs, 'actual':loss_actual})
        final_df = pd.concat([final_df,df])

final_df.reset_index(inplace=True)
final_df.to_feather("../../results/final_all_obs_COLD_DEBUG.feather")
final_df.to_csv("../../results/final_all_obs_COLD_DEBUG.csv")

obs_sites = metadata[metadata['observed'] == True]['site_id'].values
rmse_per_site = np.array([calc_rmse(final_df[final_df['site_id']==i_d]['pred'], final_df[final_df['site_id']==i_d]['actual']) for i_d in obs_sites])
pdb.set_trace()


