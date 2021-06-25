import pandas as pd
import numpy as np
import pdb
import os

metadata = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")
# obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")
obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_062321.feather")

site_ids = np.unique(obs['site_id'].values)
print(len(site_ids), ' lakes')
# site_ids = metadata['site_id'].values #CHANGE DIS----------
n_folds = 5

combined_df = pd.DataFrame()
combined_lm = pd.DataFrame()
combined_gb = pd.DataFrame()
combined_ea = pd.DataFrame()
for k in range(n_folds): #CHANGE DIS----------------

	print("fold ",k)
	# lm_df = pd.read_feather("../../results/lm_conus_022221_fold"+str(k)+".feather")
	lm_df = pd.read_feather("../../results/lm_lagless_041321_fold"+str(k)+".feather")
	# gb_df = pd.read_feather("../../results/xgb_conus_022221_fold"+str(k)+".feather")

	if os.path.exists("../../results/xgb_lagless_041321_fold"+str(k)+".feather"):
		gb_df = pd.read_feather("../../results/xgb_lagless_041321_fold"+str(k)+".feather")
	# else:
		# gb_df = pd.read_feather("../../results/xgb_lagless_041321_fold"+str(k)+".feather")


	gb_date_df = pd.read_feather("../../results/xgb_lagless_dates_fold"+str(k)+".feather")
	# ea_df = pd.read_feather("../../results/err_est_outputs_225hid_EALSTM_fold"+str(k)+".feather")
	# ea_df = pd.read_feather("../../results/err_est_outputs_1layer256hid_2.4rmse_EALSTM_fold"+str(k)+".feather")
	# ea_df = pd.read_feather("../../results/err_est_outputs_2layer128hid_2.4rmse_EALSTM_fold"+str(k)+".feather")
	ea_df = pd.read_feather("../../results/err_est_outputs_062421_EALSTM_fold"+str(k)+".feather")
	ea_df.drop(ea_df[ea_df['Date'] < gb_date_df['Date'].min()].index,axis=0,inplace=True)
	assert (ea_df['Date'].values == gb_date_df['Date'].values).all()
	assert ea_df.shape[0] == lm_df.shape[0]
	assert ea_df.shape[0] == gb_df.shape[0]

	combined_ea = combined_ea.append(ea_df)
	combined_ea.reset_index(inplace=True,drop=True)
	combined_gb = combined_gb.append(gb_df)
	combined_gb.reset_index(inplace=True,drop=True)
	combined_lm = combined_lm.append(lm_df)
	combined_lm.reset_index(inplace=True,drop=True)

combined_df['Date'] = combined_ea['Date']
combined_df['site_id'] = combined_gb['site_id']
combined_df['wtemp_predicted-ealstm'] = combined_ea['wtemp_predicted']
combined_df['wtemp_predicted-xgboost'] = combined_gb['temp_pred_xgb']
combined_df['wtemp_predicted-linear_model'] = combined_lm['temp_pred_lm']
# combined_df['wtemp_actual'] = combined_ea['wtemp_actual']
combined_df['wtemp_actual'] = combined_gb['temp_actual']
combined_df.reset_index(inplace=True)
combined_df.to_feather("../../results/all_outputs_and_obs_062321.feather")
combined_df.to_csv("../../results/all_outputs_and_obs_062321.csv")

combined_df = pd.read_feather("../../results/all_outputs_and_obs_062321.feather")

per_site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost','rmse_lm'])
# per_site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost'])
for i,site_id in enumerate(site_ids):
	print(i)
	per_site_res = combined_df[combined_df['site_id'] == site_id]
	site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost','rmse_lm'])
	# site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost'])
	site_df['rmse_ealstm'] = [np.sqrt(((per_site_res['wtemp_predicted-ealstm'] - per_site_res['wtemp_actual']) ** 2).mean())]
	site_df['rmse_xgboost'] = [np.sqrt(((per_site_res['wtemp_predicted-xgboost'] - per_site_res['wtemp_actual']) ** 2).mean())]
	site_df['rmse_lm'] = [np.sqrt(((per_site_res['wtemp_predicted-linear_model'] - per_site_res['wtemp_actual']) ** 2).mean())]
	if np.isnan(site_df['rmse_ealstm']).any():
		continue
	site_df['site_id'] = [site_id]
	site_df['n_obs'] = [per_site_res.shape[0]]
	per_site_df = per_site_df.append(site_df)

per_site_df.reset_index(inplace=True)
per_site_df.to_csv("../../results/err_per_site_062321.csv")
per_site_df.to_feather("../../results/err_per_site_062321.feather")
