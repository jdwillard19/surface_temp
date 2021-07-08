import pandas as pd
import numpy as np
import sys
import os
import pdb
import xarray as xr
site_ids = np.load("../../metadata/lakeset.npy",allow_pickle=True)


#load depth
metadata_wDepth = pd.read_feather('../../metadata/lake_metadata_MTL.feather')
# mean_feats = np.array([1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
# std_feats = np.array([8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])
base_path = '../../../lake_conus_surface_temp_2021/data/raw/data_release/'
w1 = xr.open_dataset(base_path+'01_weather_N40-53_W98-126.nc4')
dates = w1['time'].values

feat_base_path = '../../../lake_conus_surface_temp_2021/data/raw/feats/'
og_proc_base_path = '../../../lake_conus_surface_temp_2021/'


# loop to preprocess each site
for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
    # if os.path.exists("../../../lake_conus_surface_temp_2021/data/processed/"+site_id+"/features.npy"):
    #     print("already done")
    #     continue
    #get weather_id
    # w_id = metadata[metadata['site_id'] == site_id]['weather_id'].values[0].encode()


    #get and sort obs
    site_obs = pd.read_feather("../../data/raw/obs/"+site_id+"_100obs.feather")
    site_obs_trn = site_obs[site_obs['subset']=='train']
    site_obs_tst = site_obs[site_obs['subset']=='test']
    # print(site_obs.shape[0], " obs")
    site_obs_trn = site_obs_trn.sort_values("Date")    
    site_obs_tst = site_obs_tst.sort_values("Date")    

    start_date = dates[0]
    end_date = dates[-1]
    print("start date: ",start_date)
    print("end date: ", end_date)

    #cut files to between first and last observation
    lower_cutoff = np.where(dates == pd.Timestamp(start_date).to_datetime64())[0][0] #457
    print("lower cutoff: ", lower_cutoff)
    upper_cutoff = dates.shape[0]
    print("upper cutoff: ", upper_cutoff)
    site_dates = dates[lower_cutoff:upper_cutoff]
    n_dates = len(site_dates)
    print("n dates after cutoff: ", n_dates)

    
    #load feats
    og_feats_raw = np.load(og_proc_base_path+'data/processed/'+site_id+'/features.npy')
    og_feats = np.load(og_proc_base_path+'data/processed/'+site_id+'/processed_features.npy')


    #load depth
    pdb.set_trace()

    #data structs to fill
    site_obs_trn = np.empty((n_dates))
    site_obs_tst = np.empty((n_dates))
    site_obs_trn[:] = np.nan
    site_obs_tst[:] = np.nan




    # get unique observation days
    unq_obs_dates_trn = np.unique(site_obs_trn['Date'].values)
    unq_obs_dates_tst = np.unique(site_obs_tst['Date'].values)
    n_obs_trn = unq_obs_dates_trn.shape[0]
    n_obs_tst = unq_obs_dates_tst.shape[0]
    n_obs_placed_trn = 0
    n_obs_placed_tst = 0
    for o in range(0,n_obs_trn):
        if len(np.where(dates == pd.Timestamp(site_obs_trn['Date'].values[o]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            pdb.set_trace() #deprecated?
            obs_d += 1
            continue
        date_ind = np.where(dates == pd.Timestamp(site_obs_trn['Date'].values[o]).to_datetime64())[0][0]
        site_obs_trn[date_ind] = site_obs_trn['wtemp_obs'].values[o]
        n_obs_placed_trn += 1
    for o in range(0,n_obs_tst):
        if len(np.where(dates == pd.Timestamp(site_obs_tst['Date'].values[o]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            pdb.set_trace() #deprecated?
            obs_d += 1
            continue
        date_ind = np.where(dates == pd.Timestamp(site_obs_tst['Date'].values[o]).to_datetime64())[0][0]
        site_obs_tst[date_ind] = site_obs_tst['wtemp_obs'].values[o]
        n_obs_placed_tst += 1

    assert np.count_nonzero(site_obs_trn) == 70
    assert np.count_nonzero(site_obs_tst) == 30

    #make directory if not exist
    if not os.path.exists("../../data/processed/"+site_id): 
        os.mkdir("../../data/processed/"+site_id)
    if not os.path.exists("../../models/"+site_id):
        os.mkdir("../../models/"+site_id)

    feat_path = "../../data/processed/"+site_id+"/features"
    norm_feat_path = "../../data/processed/"+site_id+"/processed_features"
    obs_path_trn = "../../data/processed/"+site_id+"/trn"
    obs_path_tst = "../../data/processed/"+site_id+"/tst"
    dates_path = "../../data/processed/"+site_id+"/dates"

    #assert and save
    assert np.isfinite(site_feats).all()
    assert np.isfinite(feats_norm).all()
    assert np.isfinite(dates).all()

    np.save(feat_path, site_feats)
    np.save(norm_feat_path, feats_norm)
    np.save(dates_path, dates)
    np.save(obs_path_trn, site_obs_trn)
    np.save(obs_path_tst, site_obs_tst)
