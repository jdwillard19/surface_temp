import pandas as pd
import numpy as np
import pdb

metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")
obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")
site_ids = metadata['site_id'].values
n_folds = 5

combined_df = pd.DataFrame()
combined_lm = pd.DataFrame()
combined_gb = pd.DataFrame()
combined_ea = pd.DataFrame()
for k in range(n_folds):


	lm_df = pd.read_feather("../../results/lm_conus_022221_fold"+str(k)+".feather")
	gb_df = pd.read_feather("../../results/xgb_conus_022221_fold"+str(k)+".feather")
	gb_date_df = pd.read_feather("../../results/xgb_dates_conus_022221_fold"+str(k)+".feather")
	ea_df = pd.read_feather("../../results/err_est_outputs_EALSTM_fold"+str(k)+".feather")

	ind_to_remove = []
	rm_ct = 0
	skipped = False
	for i,date in enumerate(ea_df['Date'].values):
		while gb_date_df.iloc[i+rm_ct,2] != ea_df.iloc[i,0]:
			pdb.set_trace()
			rm_ct += 1
			ind_to_remove.append(i)
	lm_df.drop(ind_to_remove)
	gb_df.drop(ind_to_remove)
	gb_date_df.drop(ind_to_remove)
	# ind_to_remove = []
	# for i,date in enumerate(ea_df['Date'].values):
	# 	if gb_date_df.iloc[i,2] is not ea_df.iloc[i,0]:
	# 		ind_to_remove.append
	pdb.set_trace()