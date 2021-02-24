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
	ea_df = pd.read_feather("../../results/err_est_outputs_EALSTM_fold"+str(k)+".feather")

	pdb.set_trace()