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
# metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_file_020421.csv")

#get site ids
site_ids = np.unique(metadata['site_id'].values)
n_lakes = site_ids.shape[0]

#load wst obs
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")



n_dyn_feats = 5 #AT,LW,SW,WSU,WSV
n_stc_feats = 4 #AREA,LAT,LON,ELEV
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
        x = str(metadata[metadata['site_id'] == name]['x'].values[0])+".0"
        y = str(metadata[metadata['site_id'] == name]['y'].values[0])+".0"
        sw_vals = np.load("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y.npy")
        lw_vals = np.load("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y.npy")
        at_vals = np.load("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y.npy")
        wsu_vals = np.load("../../data/raw/feats/WSU_"+str(x)+"x_"+str(y)+"y.npy")
        wsv_vals = np.load("../../data/raw/feats/WSV_"+str(x)+"x_"+str(y)+"y.npy")
        means_per_lake[lake_ind,0] = sw_vals.mean()
        means_per_lake[lake_ind,1] = lw_vals.mean()
        means_per_lake[lake_ind,2] = at_vals.mean()
        means_per_lake[lake_ind,3] = wsu_vals.mean()
        means_per_lake[lake_ind,4] = wsv_vals.mean()
        var_per_lake[lake_ind,0] = sw_vals.std()
        var_per_lake[lake_ind,1] = lw_vals.std()
        var_per_lake[lake_ind,2] = at_vals.std()
        var_per_lake[lake_ind,3] = wsu_vals.std()
        var_per_lake[lake_ind,4] = wsv_vals.std()
        if not np.isfinite(means_per_lake[lake_ind,:]).all():
            pdb.set_trace()
            assert np.isfinite(means_per_lake[lake_ind,:]).all()
        if not np.isfinite(var_per_lake[lake_ind,:]).all():
            pdb.set_trace()
            assert np.isfinite(var_per_lake[lake_ind,:]).all()
        stat_vals_per_lake[lake_ind,0] = np.log(metadata[metadata['site_id'] == name]['area_m2'].values[0])
        stat_vals_per_lake[lake_ind,1] = metadata[metadata['site_id'] == name]['lat'].values[0]
        stat_vals_per_lake[lake_ind,2] = metadata[metadata['site_id'] == name]['lon'].values[0]
        stat_vals_per_lake[lake_ind,3] = metadata[metadata['site_id'] == name]['Elevation'].values[0]


    mean_feats = np.average(means_per_lake, axis=0)   
    std_feats = np.average(var_per_lake ** (.5), axis=0)   
    print("mean feats: ", repr(mean_feats))
    print("std feats: ", repr(std_feats))
    # assert mean_feats.shape[0] == 8
    # assert std_feats.shape[0] == 8
    # assert not np.isnan(np.sum(mean_feats))
    # assert not np.isnan(np.sum(std_feats))
else:
    #can uncomment and hard code here 
    mean_feats = np.array([8669184.835544042, 4.09564144e+01, -9.01653002e+01,1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
    std_feats = np.array([517506195.05362266, 6.51122458e+00, 1.04199758e+01, 8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])

    #full conus
    mean_feats = np.array([1.76938519e+02, 3.07244103e+02, 2.82966424e+02, 7.85578980e-01, 2.86128260e-01,22764867.86668189,41.67704180895113,-90.42553834994683,570.7116328304598])
    std_feats = np.array([9.10541828, 7.54501692, 3.32520898, 1.6204411 , 1.70625239, 813872728.8619916,6.448248574774095,9.870393000769734,1029.6817691460385])
pdb.set_trace()
elev_ids = site_ids[np.where(np.isfinite(stat_vals_per_lake[:,3]))]
np.save("../../data/raw/static/lists/elevation_ids",elev_ids)
# sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave

# print("loading sw nc file....")
# sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
# print("sw file loaded")

# dates = sw_da['Time'].values