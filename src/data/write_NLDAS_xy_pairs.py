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
# lw_ds_path = "../../data/globus/NLDAS_DLWRFsfc_19790102-20210102_train_test.nc" #longwave
# at_ds_path = "../../data/globus/NLDAS_TMP2m_19790102-20210102_train_test.nc" #airtemp
# sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
wsu_ds_path = "../../data/globus/NLDAS_UGRD10m_19790102-20210102_train_test.nc" #windspeed u
wsv_ds_path = "../../data/globus/NLDAS_VGRD10m_19790102-20210102_train_test.nc" #windspeed v

# print("loading sw nc file....")
# sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
# print("sw file loaded")
# print("loading lw nc file....")
# lw_da = xr.open_dataset(lw_ds_path)['DLWRFsfc']
# print("lw file loaded")
# print("loading at nc file....")
# at_da = xr.open_dataset(at_ds_path)['TMP2m']
# print("at file loaded")
print("loading at wsu file....")
wsu_da = xr.open_dataset(wsu_ds_path)['UGRD10m']
print("wsu file loaded")
print("loading at nc file....")
wsv_da = xr.open_dataset(wsv_ds_path)['VGRD10m']
print("wsv file loaded")



for lake_ind, name in enumerate(site_ids):

    print("(",lake_ind,"/",str(len(site_ids)),") ","writing... ", name)

    #get NLDAS coords
    x = metadata[metadata['site_id'] == name]['x'].values[0]
    y = metadata[metadata['site_id'] == name]['y'].values[0]

    # sw_vals = sw_da[:,y,x].values
    # lw_vals = lw_da[:,y,x].values
    # at_vals = sw_da[:,y,x].values
    wsu_vals = wsu_da[:,y,x].values
    wsv_vals = wsv_da[:,y,x].values
    # np.save("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y",sw_vals)
    # np.save("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y",lw_vals)
    # np.save("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y",at_vals)
    np.save("../../data/raw/feats/WSU_"+str(x)+"x_"+str(y)+"y",wsu_vals)
    np.save("../../data/raw/feats/WSV_"+str(x)+"x_"+str(y)+"y",wsv_vals)

