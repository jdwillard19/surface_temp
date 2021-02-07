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
site_ids = np.unique(metadata['site_id'].values)

# metadata.set_index("site_id",inplace=True)
#load wst obs
obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")
obs.sort_values('Date',inplace=True)
obs = obs[:-2] #delete error obs year 2805, and 2021

#get site ids
n_lakes = site_ids.shape[0]

#load NLDAS data
# lw_ds_path = "../../data/globus/NLDAS_DLWRFsfc_19790102-20210102_train_test.nc" #longwave
# at_ds_path = "../../data/globus/NLDAS_TMP2m_19790102-20210102_train_test.nc" #airtemp
# sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
# wsu_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
# wsv_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
#wsu_ds_path=

# lw_ds = xr.open_dataset(lw_ds_path)
# at_ds = xr.open_dataset(at_ds_path)
# sw_ds = xr.open_dataset(sw_ds_path)

n_dyn_feats = 5 #AT,LW,SW,WSU,WSV
n_stc_feats = 4 #AREA,LAT,LON,ELEV

#can uncomment and hard code here 
# mean_feats = np.array([1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
# std_feats = np.array([8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])

# mean_feats = np.array([8669184.835544042, 4.09564144e+01, -9.01653002e+01,1.78920854e+02, 3.10114323e+02, 1.04963419e+01, 7.65040625e-01, 3.06756386e-01])
# std_feats = np.array([517506195.05362266, 6.51122458e+00, 1.04199758e+01, 9.06977398, 7.51899974, 3.20886119, 1.61188199, 1.69147159])

#full conus
mean_feats = np.array([22764867.86668189,41.67704180895113,-90.42553834994683,570.7116328304598,1.76938519e+02, 3.07244103e+02, 2.82966424e+02, 7.85578980e-01, 2.86128260e-01])
std_feats = np.array([813872728.8619916,6.448248574774095,9.870393000769734,1029.6817691460385,9.10541828, 7.54501692, 3.32520898, 1.6204411 , 1.70625239])

n_features = mean_feats.shape[0]
#load dates
sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
print("loading sw nc file....")
sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
print("sw file loaded")
dates = sw_da['Time'].values


start = 0
end = len(site_ids)
date_offset = 350


for site_ct, site_id in enumerate(site_ids[start:end]):
    if site_ct < 2918:
        continue
    print(site_ct," starting ", site_id)

    #get NLDAS coords
    x = str(metadata[metadata['site_id'] == site_id]['x'].values[0])+".0"
    y = str(metadata[metadata['site_id'] == site_id]['y'].values[0])+".0"

    #read/format meteorological data for numpy
    site_obs = obs[obs['site_id'] == site_id]
    print(site_obs.shape[0], " obs")

    #lower/uppur cutoff indices (to match observations)
    if site_obs.shape[0] == 0:
        print("|\n|\nNO SURFACE OBSERVATIONS\n|\n|\n|")
        pdb.set_trace()
        no_obs_ids.append(site_id)
        no_obs_ct +=1 
        continue

    site_obs = site_obs.sort_values("Date")    
    #sort observations
    obs_start_date = site_obs.values[0,0]
    meteo_start_date = dates[0]
    start_date = None
    #do date offset for pre-pend meteo
    if pd.Timestamp(obs_start_date) - pd.DateOffset(days=date_offset) < pd.Timestamp(meteo_start_date):
        start_date = meteo_start_date
    else:
        start_date = str(pd.Timestamp(obs_start_date) - pd.DateOffset(days=date_offset))[:10]

    obs_end_date = site_obs.values[-1,0]
    # meteo_end_date = dates[-1]

    print("start date: ",start_date)
    print("end date: ", obs_end_date)
    #cut files to between first and last observation
    lower_cutoff = np.where(dates == pd.Timestamp(start_date).to_datetime64())[0][0] #457
    print("lower cutoff: ", lower_cutoff)
    if len(np.where(dates == pd.Timestamp(obs_end_date).to_datetime64())[0]) < 1: 
        print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
        upper_cutoff = dates.shape[0]
    else:
        upper_cutoff = np.where(dates == pd.Timestamp(obs_end_date).to_datetime64())[0][0]+1 #14233
    print("upper cutoff: ", upper_cutoff)
    site_dates = dates[lower_cutoff:upper_cutoff]
    n_dates = len(site_dates)
    print("n dates after cutoff: ", n_dates)
   # #cut files to between first and last GLM simulated date for pre-train data
   #  if len(np.where(meteo_dates_pt == start_date_pt)[0]) < 1: 
   #      print("observation beyond meteorological data! PRE-TRAIN data will only be starting at the start of meteorological data")
   #      start_date_pt = meteo_dates_pt[0]
   #  if len(np.where(meteo_dates_pt == end_date_pt)[0]) < 1: 
   #      print("observation beyond meteorological data! PRE-TRAIN data will only be used up to the end of meteorological data")
   #      end_date_pt = meteo_dates_pt[-1]
   #  lower_cutoff_pt = np.where(meteo_dates_pt == start_date_pt)[0][0] #457
   #  upper_cutoff_pt = np.where(meteo_dates_pt == end_date_pt)[0][0] #457
   #  meteo_dates_pt = meteo_dates_pt[lower_cutoff_pt:upper_cutoff_pt]
    


    



    #read from file and filter dates
    # meteo = np.genfromtxt(base_path+'meteo/nhdhr_'+site_id+'_meteo.csv', delimiter=',', usecols=(3,4,5,6,7,8,9), skip_header=1)
    # meteo_pt = np.genfromtxt(base_path+'meteo/nhdhr_'+site_id+'_meteo.csv', delimiter=',', usecols=(3,4,5,6,7,8,9), skip_header=1)
    # meteo = meteo[lower_cutoff:upper_cutoff,:]
    # meteo_pt = meteo_pt[lower_cutoff_pt:upper_cutoff_pt,:]

    site_feats = np.empty((n_dates,n_features))
    site_feats[:] = np.nan
    sw = np.load("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y.npy")
    lw = np.load("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y.npy")
    at = np.load("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y.npy")
    wsu = np.load("../../data/raw/feats/WSU_"+str(x)+"x_"+str(y)+"y.npy")
    wsv = np.load("../../data/raw/feats/WSV_"+str(x)+"x_"+str(y)+"y.npy")

    site_feats[:,0] = metadata[metadata['site_id']==site_id].area_m2
    site_feats[:,1] = metadata[metadata['site_id']==site_id].lat
    site_feats[:,2] = metadata[metadata['site_id']==site_id].lon
    site_feats[:,3] = metadata[metadata['site_id']==site_id].Elevation
    pdb.set_trace()
    site_feats[:,4] = sw[lower_cutoff:upper_cutoff]
    site_feats[:,5] = lw[lower_cutoff:upper_cutoff]
    site_feats[:,6] = at[lower_cutoff:upper_cutoff]
    site_feats[:,7] = wsu[lower_cutoff:upper_cutoff]
    site_feats[:,8] = wsv[lower_cutoff:upper_cutoff]

    #normalize data
    feats_norm = (site_feats - mean_feats[:]) / std_feats[:]


    obs_trn_mat = np.empty((n_dates))
    site_obs_mat = np.empty((n_dates))
    site_obs_mat[:] = np.nan
    obs_trn_mat[:] = np.nan
    obs_tst_mat = np.empty((n_dates))
    obs_tst_mat[:] = np.nan

    # feat_mat_pt[:,:-1] = meteo_pt[:]
    # feat_mat_pt[:,-1] = ice_flags_pt[:,1]        
    # feat_norm_mat_pt[:,:] = meteo_norm_pt[:]
    # glm_mat_pt[:] = glm_temps_pt[:]

    #fill train data
    # feat_mat[:,:-n_static_feats-1] = meteo[:]
    # feat_mat[:,-n_static_feats-1] = ice_flags[:,1]        
    # feat_norm_mat[:,:n_features] = meteo_norm[:]
    # glm_mat[:] = glm_temps[:]

    #verify all mats filled

    # if np.isnan(np.sum(feats_norm)):
    #     raise Exception("ERROR: Preprocessing failed, there is missing data feat norm")
    #     sys.exit() 
    # if np.isnan(np.sum(site_feats)):
    #     raise Exception("ERROR: Preprocessing failed, there is missing data feat ")
    #     sys.exit() 


    obs_g = 0
    obs_d = 0

    #get unique observation days
    unq_obs_dates = np.unique(site_obs.values[:,0])
    n_unq_obs_dates = unq_obs_dates.shape[0]
    n_obs = n_unq_obs_dates

    #place obs data
    n_obs_placed = 0
    # n_trn_obs_placed = 0
    for o in range(0,n_obs):
        if len(np.where(site_dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            pdb.set_trace()
            obs_d += 1
            continue
        date_ind = np.where(site_dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0][0]
        site_obs_mat[date_ind] = site_obs.values[o,2]
        n_obs_placed += 1


    if not os.path.exists("../../data/processed/"+site_id): 
        os.mkdir("../../data/processed/"+site_id)
    if not os.path.exists("../../models/"+site_id):
        os.mkdir("../../models/"+site_id)

    feat_path = "../../data/processed/"+site_id+"/features_ea_conus"
    norm_feat_path = "../../data/processed/"+site_id+"/processed_features_ea_conus"
    full_path = "../../data/processed/"+site_id+"/full"
    dates_path = "../../data/processed/"+site_id+"/dates"


    np.save(feat_path, site_feats)
    np.save(norm_feat_path, feats_norm)
    np.save(dates_path, site_dates)
    np.save(full_path, site_obs_mat)

