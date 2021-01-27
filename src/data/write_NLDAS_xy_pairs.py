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
# metadata2 = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_file_temp.csv")
# pdb.set_trace()
#load wst obs
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")

#get site ids
site_ids = np.unique(metadata['site_id'].values)
n_lakes = site_ids.shape[0]

#load NLDAS data
lw_ds_path = "../../data/globus/NLDAS_DLWRFsfc_19790102-20210102_train_test.nc" #longwave
at_ds_path = "../../data/globus/NLDAS_TMP2m_19790102-20210102_train_test.nc" #airtemp
sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
wsu_ds_path = "../../data/globus/NLDAS_UGRD10m_19790102-20210102_train_test.nc" #windspeed u
wsv_ds_path = "../../data/globus/NLDAS_VGRD10m_19790102-20210102_train_test.nc" #windspeed v

print("loading sw nc file....")
sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
print("sw file loaded")
print("loading lw nc file....")
lw_da = xr.open_dataset(lw_ds_path)['DLWRFsfc']
print("lw file loaded")
print("loading at nc file....")
at_da = xr.open_dataset(at_ds_path)['TMP2m']
print("at file loaded")
print("loading at wsu file....")
wsu_da = xr.open_dataset(wsu_ds_path)['UGRD10m']
print("wsu file loaded")
print("loading at nc file....")
wsv_da = xr.open_dataset(wsv_ds_path)['VGRD10m']
print("wsv file loaded")

start = int(sys.argv[1])
end = int(sys.argv[2])
# site_ids = np.flipud(site_ids)
print("running site id's ",start,"->",end)
site_ids = site_ids[start:end]
for lake_ind, name in enumerate(site_ids):
    # if lake_ind < 565:
    #     continue
    # print("(",len(site_ids)-lake_ind,"/",str(len(site_ids)),") ","writing... ", name)
    print("(",lake_ind,"/",str(len(site_ids)),") ","writing... ", name)

    #get NLDAS coords
    x = metadata[metadata['site_id'] == name]['x'].values[0]-1
    y = metadata[metadata['site_id'] == name]['y'].values[0]-1
    # if os.path.exists("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y"):
    #     continue
    sw_vals = sw_da[:,y,x].values
    lw_vals = lw_da[:,y,x].values
    at_vals = at_da[:,y,x].values
    wsu_vals = wsu_da[:,y,x].values
    wsv_vals = wsv_da[:,y,x].values
    if np.isnan(sw_vals).any():
        print("nan sw?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(lw_vals).any():
        print("nan lw?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(at_vals).any():
        print("nan at?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(wsu_vals).any():
        print("nan wsu?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(wsv_vals).any():
        print("nan wsv?") 
        raise Exception("CANT CONTINUE") 

    np.save("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y",sw_vals)
    np.save("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y",lw_vals)
    np.save("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y",at_vals)
    np.save("../../data/raw/feats/WSU_"+str(x)+"x_"+str(y)+"y",wsu_vals)
    np.save("../../data/raw/feats/WSV_"+str(x)+"x_"+str(y)+"y",wsv_vals)
    print("x/y: ",x,"/",y,":\nSW: ", sw_vals, "\nLW: ",lw_vals,"\nAT: ",at_vals,"\nWSU: ", wsu_vals, "\nWSV: ", wsv_vals)

print("DATA COMPLETE")