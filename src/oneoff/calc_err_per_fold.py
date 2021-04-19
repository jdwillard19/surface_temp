import pandas as pd
import numpy as np
import pdb
import os

err_df = pd.read_feather("../../results/err_per_site_041921.feather")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")

err_per_fold = np.empty((5))
err_per_fold[:] = np.nan
for k in range(5):
	err_per_fold[k] = np.median(err_df[np.isin(err_df['site_id'],metadata[metadata['5fold_fold']==k]['rmse_ealstm'].values)

print(err_per_fold)


