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
metadata = pd.read_csv("../../metadata/surface_lake_metadata_file_temp.csv")
site_ids = np.unique(metadata['site_id'].values)

metadata.set_index("site_id",inplace=True)
#load wst obs
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")
obs = obs[:-1] #delete error obs

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
n_stc_feats = 3 #AREA,LAT,LON

#can uncomment and hard code here 
# mean_feats = np.array([1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
# std_feats = np.array([8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])

mean_feats = np.array([8669184.835544042, 4.09564144e+01, -9.01653002e+01,1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
std_feats = np.array([517506195.05362266, 6.51122458e+00, 1.04199758e+01, 8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])
n_features = mean_feats.shape[0]
#load dates
sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
print("loading sw nc file....")
sw_da = xr.open_dataset(sw_ds_path)['DSWRFsfc']
print("sw file loaded")
dates = sw_da['Time'].values


start = 0
end = len(site_ids)



for site_ct, site_id in enumerate(site_ids[start:end]):

    print(site_ct," starting ", site_id)

    #get NLDAS coords
    x = int(metadata.loc[site_id]['x'])-1
    y = int(metadata.loc[site_id]['y'])-1

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
    if pd.Timestamp(obs_start_date) - pd.DateOffset(days=90) < pd.Timestamp(meteo_start_date):
        start_date = meteo_start_date
    else:
        start_date = str(pd.Timestamp(obs_start_date) - pd.DateOffset(days=90))[:10]

    obs_end_date = site_obs.values[-1,0]
    # meteo_end_date = dates[-1]

    print("start date: ",start_date)
    print("end date: ", obs_end_date)

    #cut files to between first and last observation
    lower_cutoff = np.where(dates == pd.Timestamp(start_date).to_datetime64())[0][0] #457
    print("lower cutoff: ", lower_cutoff)
    if len(np.where(dates == obs_end_date)[0]) < 1: 
        print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
        upper_cutoff = dates.shape[0]
    else:
        upper_cutoff = np.where(dates == obs_end_date)[0][0]+1 #14233
    print("upper cutoff: ", upper_cutoff)
    dates = dates[lower_cutoff:upper_cutoff]
    n_dates = len(dates)
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

    site_feats[:,0] = sw[lower_cutoff:upper_cutoff]
    site_feats[:,1] = lw[lower_cutoff:upper_cutoff]
    site_feats[:,2] = at[lower_cutoff:upper_cutoff]
    site_feats[:,3] = wsu[lower_cutoff:upper_cutoff]
    site_feats[:,4] = wsv[lower_cutoff:upper_cutoff]
    site_feats[:,5] = metadata.loc[site_id].area_m2
    site_feats[:,6] = metadata.loc[site_id].lat
    site_feats[:,7] = metadata.loc[site_id].lon
    pdb.set_trace()
    #normalize data
    meteo_norm = (meteo - mean_feats[:]) / std_feats[:]

    ################################################################################
    # read/format GLM temperatures and observation data for numpy
    ###################################################################################
    n_total_dates = dates.shape[0]


    #cut glm temps to meteo dates for observation data PRETRAIN
    if len(np.where(glm_temps_pt[:,-1] == start_date_pt[0])) < 1:
        print("pretrain glm outputs begin at " + start_date_pt + "which is before GLM data which begins at " + glm_temps_pt[0,0])
        lower_cutoff_pt = 0
        new_meteo_lower_cutoff_pt = np.where(meteo_dates_pt == glm_temps_pt[0,-1])[0][0]
        meteo_pt = meteo_pt[new_meteo_lower_cutoff_pt:,:]
        meteo_norm_pt = meteo_norm_pt[new_meteo_lower_cutoff_pt:,:]
        meteo_dates_pt = meteo_dates_pt[new_meteo_lower_cutoff_pt:]
    else:
        lower_cutoff_pt = np.where(glm_temps_pt[:,-1] == start_date_pt)[0][0] 

    if len(np.where(glm_temps_pt[:,-1] == end_date_pt)[0]) < 1: 
        print("pretrain glm outputs extend to " + end_date_pt + "which is beyond GLM data which extends to " + glm_temps_pt[-1,-1])
        upper_cutoff_pt = glm_temps_pt[:,-1].shape[0]
        new_meteo_upper_cutoff_pt = np.where(meteo_dates_pt == glm_temps_pt[-1,-1])[0][0]
        meteo_pt = meteo_pt[:new_meteo_upper_cutoff_pt,:]
        meteo_norm_pt = meteo_norm_pt[:new_meteo_upper_cutoff_pt,:]
        meteo_dates_pt = meteo_dates_pt[:new_meteo_upper_cutoff_pt]
    else:
        upper_cutoff_pt = np.where(glm_temps_pt[:,-1] == end_date_pt)[0][0] 

    glm_temps_pt = glm_temps_pt[lower_cutoff_pt:upper_cutoff_pt,:]
    ice_flags_pt = ice_flags_pt[lower_cutoff_pt:upper_cutoff_pt,:]

    #cut glm temps to meteo dates for observation data
    if len(np.where(glm_temps[:,-1] == start_date)[0]) < 1:
        print("observations begin at " + start_date + "which is before GLM data which begins at " + glm_temps[0,-1])
        lower_cutoff = 0
        new_meteo_lower_cutoff = np.where(meteo_dates == glm_temps[0,-1])[0][0]
        meteo = meteo[new_meteo_lower_cutoff:,:]
        meteo_norm = meteo_norm[new_meteo_lower_cutoff:,:]
        meteo_dates = meteo_dates[new_meteo_lower_cutoff:]
    else:
        lower_cutoff = np.where(glm_temps[:,-1] == start_date)[0][0] 

    if len(np.where(glm_temps[:,-1] == end_date)[0]) < 1: 
        print("observations extend to " + end_date + "which is beyond GLM data which extends to " + glm_temps[-1,-1])
        upper_cutoff = glm_temps[:,-1].shape[0]
        new_meteo_upper_cutoff = np.where(meteo_dates == glm_temps[-1,-1])[0][0] + 1
        meteo = meteo[:new_meteo_upper_cutoff,:]
        meteo_norm = meteo_norm[:new_meteo_upper_cutoff,:]
        meteo_dates = meteo_dates[:new_meteo_upper_cutoff]
    else:
        upper_cutoff = np.where(glm_temps[:,-1] == end_date)[0][0] +1

    glm_temps = glm_temps[lower_cutoff:upper_cutoff,:]
    ice_flags = ice_flags[lower_cutoff:upper_cutoff,:]
    # n_dates_pt = glm_temps_pt.shape[0]
    n_dates = glm_temps.shape[0]
    # print("dates: ", n_dates, " , pretrain dates: ", n_dates_pt)



    # print("pretrain dates: ", start_date_pt, "->", end_date_pt)
    print("train dates: ", start_date, "->", end_date)
    ############################################################
    #fill numpy matrices
    ##################################################################

    #list static feats fo EA-LSTM
    static_feats = ['surface_area','SDF','K_d','longitude','latitude',\
                    'lw_std',\
                    'sw_mean', 'sw_std_sp','sw_mean_au',
                    'at_std_au','at_mean_au',\
                    'rh_mean_su','rh_mean_au',\
                    'rain_mean_au',\
                    'snow_mean_au']
    n_static_feats = len(static_feats)
    # feat_mat_pt = np.empty((n_dates_pt, n_features+1+n_static_feats)) #[7 meteo features-> ice flag]
    # feat_mat_pt[:] = np.nan
    # feat_norm_mat_pt = np.empty((n_dates_pt, n_features+n_static_feats)) #[7 std meteo features]
    # feat_norm_mat_pt[:] = np.nan
    # glm_mat_pt = np.empty((n_dates_pt))
    # glm_mat_pt[:] = np.nan


    feat_mat = np.empty((n_dates, n_features+1+n_static_feats)) #[7 meteo features-> ice flag]
    feat_mat[:] = np.nan
    feat_norm_mat = np.empty((n_dates, n_features+n_static_feats)) #[7 std meteo features]
    feat_norm_mat[:] = np.nan
    # glm_mat = np.empty((n_dates))
    # glm_mat[:] = np.nan
    # obs_trn_mat = np.empty((n_dates))
    # obs_trn_mat[:] = np.nan
    # obs_tst_mat = np.empty((n_dates))
    # obs_tst_mat[:] = np.nan

    # feat_mat_pt[:,:-1] = meteo_pt[:]
    # feat_mat_pt[:,-1] = ice_flags_pt[:,1]        
    # feat_norm_mat_pt[:,:] = meteo_norm_pt[:]
    # glm_mat_pt[:] = glm_temps_pt[:]

    #fill train data
    feat_mat[:,:-n_static_feats-1] = meteo[:]
    feat_mat[:,-n_static_feats-1] = ice_flags[:,1]        
    feat_norm_mat[:,:n_features] = meteo_norm[:]
    # glm_mat[:] = glm_temps[:]

    #verify all mats filled

    # if np.isnan(np.sum(feat_norm_mat)):
    #     raise Exception("ERROR: Preprocessing failed, there is missing data feat norm")
    #     sys.exit() 


    # obs_g = 0
    # obs_d = 0

    #get unique observation days
    # unq_obs_dates = np.unique(obs[:,0])
    # n_unq_obs_dates = unq_obs_dates.shape
    # first_tst_date = obs[0,0]
    # last_tst_date = obs[math.floor(obs.shape[0]/3),0]
    # last_tst_obs_ind = np.where(obs[:,0] == last_tst_date)[0][-1]
    # n_pretrain = meteo_dates_pt.shape[0]
    # n_tst = last_tst_obs_ind + 1
    # n_trn = obs.shape[0] - n_tst

    # last_train_date = obs[-1,0]
    # if last_tst_obs_ind + 1 >= obs.shape[0]:
    #     last_tst_obs_ind -= 1
    # first_train_date = obs[last_tst_obs_ind + 1,0]
    # first_pretrain_date = meteo_dates_pt[0]
    # last_pretrain_date = meteo_dates_pt[-1]
   
    #test data
    # n_tst_obs_placed = 0
    # n_trn_obs_placed = 0
    # for o in range(0,last_tst_obs_ind+1):
    #     if len(np.where(meteo_dates == obs[o,0])[0]) < 1:
    #         # print("not within meteo dates")
    #         obs_d += 1
    #         continue
    #     date_ind = np.where(meteo_dates == obs[o,0])[0][0]
    #     obs_tst_mat[date_ind] = obs[o,2]
    #     n_tst_obs_placed += 1

    # #train data
    # for o in range(last_tst_obs_ind+1, n_obs):
    #     if len(np.where(meteo_dates == obs[o,0])[0]) < 1:
    #         obs_d += 1
    #         continue

    #     date_ind = np.where(meteo_dates == obs[o,0])[0][0]

    #     obs_trn_mat[date_ind] = obs[o,2]
    #     n_trn_obs_placed += 1


    # d_str = ""
    # if obs_d > 0:
    #     d_str = ", and "+str(obs_d) + " observations outside of combined date range of meteorological and GLM output"
    # # if obs_g > 0 or obs_d > 0:
    #     # continue



    #add static features
    static_feat_means = metadata[static_feats].mean(axis=0)
    static_feat_std = metadata[static_feats].std(axis=0)
    # for feat in static_feats:
    feat_mat[:,-n_static_feats:] = metadata[metadata['site_id']==site_id][static_feats]
    feat_norm_mat[:,-n_static_feats:] = (metadata[metadata['site_id']==site_id][static_feats] - static_feat_means) / static_feat_std
    # pdb.set_trace()



    if np.isnan(np.sum(feat_mat)):
        raise Exception("ERROR: Preprocessing failed, there is missing data: features for training")
        sys.exit()
    # if np.isnan(np.sum(feat_mat_pt)):
    #     raise Exception("ERROR: Preprocessing failed, there is missing data: features for pretraining ")
    #     sys.exit()
    if np.isnan(np.sum(feat_norm_mat)):
        raise Exception("ERROR: Preprocessing failed, there is missing data feat norm")
        sys.exit() 
    #remember add sqrt surfarea
    #write features and labels to processed data
    # print("pre-training: ", first_pretrain_date, "->", last_pretrain_date, "(", n_pretrain, ")")
    # print("training: ", first_train_date, "->", last_train_date, "(", n_trn, ")")
    # print("testing: ", first_tst_date, "->", last_tst_date, "(", n_tst, ")")
    # if not os.path.exists("../../data/processed/"+site_id): 
        # os.mkdir("../../data/processed/"+site_id)
    # if not os.path.exists("../../models/"+site_id):
        # os.mkdir("../../models/"+site_id)
    # feat_path_pt = "../../data/processed/"+site_id+"/features_pt"
    feat_path = "../../data/processed/"+site_id+"/features_ea"
    # norm_feat_path_pt = "../../data/processed/"+site_id+"/processed_features_pt"
    norm_feat_path = "../../data/processed/"+site_id+"/processed_features_ea"
    # glm_path_pt = "../../data/processed/"+site_id+"/glm_pt"
    # glm_path = "../../data/processed/"+site_id+"/glm"
    # trn_path = "../../data/processed/"+site_id+"/train"
    # tst_path = "../../data/processed/"+site_id+"/test"
    # full_path = "../../data/processed/"+site_id+"/full"
    # dates_path = "../../data/processed/"+site_id+"/dates"
    # dates_path_pt = "../../data/processed/"+site_id+"/dates_pt"



    np.save(feat_path, feat_mat)
    # np.save(feat_path_pt, feat_mat_pt)
    np.save(norm_feat_path, feat_norm_mat)
    # np.save(norm_feat_path_pt, feat_norm_mat_pt)
    # np.save(glm_path, glm_mat)
    # np.save(glm_path_pt, glm_mat_pt)
    # np.save(dates_path, meteo_dates)
    # np.save(dates_path_pt, meteo_dates_pt)
    # np.save(trn_path, obs_trn_mat)
    # np.save(tst_path, obs_tst_mat)
    # full = obs_trn_mat
    # full[np.nonzero(np.isfinite(obs_tst_mat))] = obs_tst_mat[np.isfinite(obs_tst_mat)]
    # np.save(full_path, full)
    # n_obs_per.append(np.count_nonzero(np.isfinite(full)))
    # print("completed!")

# print(no_obs_ids)
# print(repr(n_obs_per))

