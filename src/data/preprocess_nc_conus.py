import xarray as xr
import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import math
import shutil
from scipy import interpolate
import pdb
import datetime



#load metadata, get ids
metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")

#load wst obs
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")

#get site ids
site_ids = np.unique(obs['site_id'].values)
n_lakes = site_ids.shape[0]

#load NLDAS data
lw_ds_path = "../../data/globus/NLDAS_DLWRFsfc_19790102-20210102_train_test.nc" #longwave
at_ds_path = "../../data/globus/NLDAS_TMP2m_19790102-20210102_train_test.nc" #airtemp
sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
wsu_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
wsv_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
#wsu_ds_path=

# lw_ds = xr.open_dataset(lw_ds_path)
# at_ds = xr.open_dataset(at_ds_path)
# sw_ds = xr.open_dataset(sw_ds_path)
print("loading sw nc file....")
sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
print("sw file loaded")
print("loading lw nc file....")
lw_da = xr.open_dataset(lw_ds_path)['DLWRFsfc']
print("lw file loaded")
print("loading at nc file....")
at_da = xr.open_dataset(at_ds_path)['TMP2m']
print("at file loaded")

dates = sw_da['Time'].values
n_dyn_feats = 3 #AT,LW,SW
n_stc_feats = 3 #AREA,LAT,LON
means_per_lake = np.zeros((n_lakes,n_dyn_feats), dtype=np.float_)
means_per_lake[:] = np.nan
var_per_lake = np.zeros((n_lakes,n_dyn_feats),dtype=np.float_)
var_per_lake[:] = np.nan
stat_vals_per_lake = np.zeros((n_lakes,n_stc_feats),dtype=np.float_)
stat_vals_per_lake[:] = np.nan

hardcode = False
if not hardcode:
    for lake_ind, name in enumerate(site_ids):

        print("(",lake_ind,"/",str(len(site_ids)),") ","pre ", name)

        #get NLDAS coords
        x = metadata[metadata['site_id'] == name]['x'].values[0]
        y = metadata[metadata['site_id'] == name]['y'].values[0]
        sw_vals = np.load("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y.npy")
        lw_vals = np.load("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y.npy")
        at_vals = np.load("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y.npy")

        means_per_lake[lake_ind,0] = sw_vals.mean()
        means_per_lake[lake_ind,1] = lw_vals.mean()
        means_per_lake[lake_ind,2] = at_vals.mean()
        var_per_lake[lake_ind,0] = sw_vals.std()
        var_per_lake[lake_ind,1] = lw_vals.std()
        var_per_lake[lake_ind,2] = at_vals.std()
        pdb.set_trace()



    mean_feats = np.average(means_per_lake, axis=0)   
    std_feats = np.average(var_per_lake ** (.5), axis=0)   
    print("mean feats: ", repr(mean_feats))
    print("std feats: ", repr(std_feats))
    assert mean_feats.shape[0] == 8
    assert std_feats.shape[0] == 8
    assert not np.isnan(np.sum(mean_feats))
    assert not np.isnan(np.sum(std_feats))
else:
	#can uncomment and hard code here 
	mean_feats = np.array([1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
	std_feats = np.array([8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])