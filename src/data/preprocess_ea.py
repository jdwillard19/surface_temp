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
###################################################################################
# June 2020 - preprocess for MTL publication (Jared)
# Sept 2020 - cleaned up for repo construction (Jared)
###################################################################################

#inital data load, site ids
base_path = "../../data/raw/sb_mtl_data_release/"
obs_df = pd.read_csv(base_path+"obs/temperature_observations.csv")
metadata = pd.read_feather("../../metadata/lake_metadata_surf.feather")
ids = np.unique(obs_df['site_id'].values)
ids = np.array([re.search('nhdhr_(.*)', x).group(1) for x in ids])
no_obs_ct = 0
n_features = 7
n_obs_per = []
no_obs_ids = []
n_lakes = ids.shape[0]

#accumulation data structs for averaging
means_per_lake = np.zeros((n_lakes,8), dtype=np.float_)
means_per_lake[:] = np.nan
var_per_lake = np.zeros((n_lakes,8),dtype=np.float_)
var_per_lake[:] = np.nan

max_surface_depth = 0.25
#calculate averages and std_dev for each input driver across all lakes
hardcode = True
if not hardcode:
    for lake_ind, name in enumerate(ids):
        nid = 'nhdhr_' + name

        print("(",lake_ind,"/",str(len(ids)),") ","pre ", name)

        #read/format meteorological data for numpy
        meteo_dates = np.loadtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=2)[1:]
        print("first meteo date", meteo_dates[0])


        obs = pd.read_feather(base_path+'obs/nhdhr_'+name+'_obs.feather')
        obs['date2'] = pd.to_datetime(obs.date)
        obs.sort_values('date2', inplace=True)
        print("first obs date: ",obs['date2'].values[0])

        #lower/uppur cutoff indices (to match observations)

        start_date = []
        end_date = []
        try:
            start_date = "{:%Y-%m-%d}".format(obs.values[0,1])
        except:
            start_date = obs.values[0,1]
        try:
            end_date = "{:%Y-%m-%d}".format(obs.values[-1,1])
        except:
            end_date = obs.values[-1,1]

        start_date = start_date.encode()
        end_date = end_date.encode()
        lower_cutoff = np.where(meteo_dates == start_date)[0][0] #457
        if len(np.where(meteo_dates == end_date)[0]) < 1: 
            print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
            upper_cutoff = meteo_dates.shape[0]
        else:
            upper_cutoff = np.where(meteo_dates == end_date)[0][0]+1 #14233

        meteo_dates = meteo_dates[lower_cutoff:upper_cutoff]


        #read from file and filter dates
        meteo = np.genfromtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', usecols=(3,4,5,6,7,8,9), skip_header=1)
        meteo = meteo[lower_cutoff:upper_cutoff,:]
        means_per_lake[lake_ind,1:] = [meteo[:,a].mean() for a in range(n_features)]
        var_per_lake[lake_ind,1:] = [meteo[:,a].std() ** 2 for a in range(n_features)]

        glm_temps = pd.read_csv(base_path+'predictions/pb0_nhdhr_'+name+'_temperatures.csv')
        glm_temps = glm_temps.values[:]
        n_total_dates = glm_temps.shape[0]


    mean_feats = np.average(means_per_lake, axis=0)   
    std_feats = np.average(var_per_lake ** (.5), axis=0)   
    print("mean feats: ", repr(mean_feats))
    print("std feats: ", repr(std_feats))
    assert mean_feats.shape[0] == 8
    assert std_feats.shape[0] == 8
    assert not np.isnan(np.sum(mean_feats))
    assert not np.isnan(np.sum(std_feats))


#can uncomment and hard code here 
mean_feats = np.array([1.66308346e02, 2.91540662e02, 6.68199233e00, 7.37268070e01, 4.79260805e00, 1.81936454e-03, 2.30189504e-03])
std_feats = np.array([8.52790273e+01, 6.10175316e+01, 1.28183124e+01, 1.29724391e+01, 1.69513213e+00, 5.54588726e-03, 1.27910016e-02])
#now preprocess every lakes data for modeling
for it_ct,nid in enumerate(ids): #for each new additional lake
    name = str(nid)
    nid = 'nhdhr_' + str(nid)

    print(it_ct," starting ", name)

    #read/format meteorological data for numpy
    meteo_dates = np.loadtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=2, skiprows=1)
    meteo_dates = np.array([x.decode() for x in meteo_dates])
    meteo_dates_pt = np.loadtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=2, skiprows=1)
    meteo_dates_pt = np.array([x.decode() for x in meteo_dates_pt])
    glm_temps = pd.read_csv(base_path+'predictions/pb0_nhdhr_'+name+'_temperatures.csv').values[:]
    # glm_temps[:,-1] = np.array([x.decode() for x in glm_temps[:,-1]])
    glm_temps_pt = pd.read_csv(base_path+'predictions/pb0_nhdhr_'+name+'_temperatures.csv').values[:]


 
    if isinstance(glm_temps[0,0], str):
        tmp = glm_temps[:,0][:]
        tmp = np.reshape(tmp,(len(tmp),1))
        glm_temps = np.delete(glm_temps, 0, 1)[:]
        glm_temps_pt = np.delete(glm_temps_pt, 0, 1)[:]
        glm_temps = np.append(glm_temps, tmp, axis=1)
        glm_temps_pt = np.append(glm_temps_pt, tmp, axis=1)

    ice_flags = pd.read_csv(base_path+ 'ice_flags/pb0_nhdhr_'+name+'_ice_flags.csv').values[:]
    ice_flags_pt = pd.read_csv(base_path + 'ice_flags/pb0_nhdhr_'+name+'_ice_flags.csv').values[:]

    #lower/uppur cutoff indices (to match observations)
    obs = pd.read_feather(base_path+'obs/nhdhr_'+name+"_obs.feather")
    obs = obs[obs['depth'] <= max_surface_depth] 
    if obs.shape[0] == 0:
        print("|\n|\nNO SURFACE OBSERVATIONS\n|\n|\n|")
        no_obs_ids.append(name)
        no_obs_ct +=1 
        continue

    obs.sort_values(by='date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False)
    
    #sort observations
    start_date = obs.values[0,1]
    start_date_pt = glm_temps[0,-1]


    #do date offset for pre-pend meteo
    if pd.Timestamp(start_date) - pd.DateOffset(days=90) < pd.Timestamp(start_date_pt):
        start_date = str(pd.Timestamp(start_date) - pd.DateOffset(days=90))[:10]
    else:
        start_date = start_date_pt

    assert start_date_pt == ice_flags_pt[0,0]
    end_date = obs.values[-1,1]
    end_date_pt = "{:%Y-%m-%d}".format(pd.Timestamp(glm_temps[-1,-1]))


    #cut files to between first and last observation
    lower_cutoff = np.where(meteo_dates == start_date)[0][0] #457
    if len(np.where(meteo_dates == end_date)[0]) < 1: 
        print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
        upper_cutoff = meteo_dates.shape[0]
    else:
        upper_cutoff = np.where(meteo_dates == end_date)[0][0]+1 #14233
    meteo_dates = meteo_dates[lower_cutoff:upper_cutoff]

   #cut files to between first and last GLM simulated date for pre-train data
    if len(np.where(meteo_dates_pt == start_date_pt)[0]) < 1: 
        print("observation beyond meteorological data! PRE-TRAIN data will only be starting at the start of meteorological data")
        start_date_pt = meteo_dates_pt[0]
    if len(np.where(meteo_dates_pt == end_date_pt)[0]) < 1: 
        print("observation beyond meteorological data! PRE-TRAIN data will only be used up to the end of meteorological data")
        end_date_pt = meteo_dates_pt[-1]
    lower_cutoff_pt = np.where(meteo_dates_pt == start_date_pt)[0][0] #457
    upper_cutoff_pt = np.where(meteo_dates_pt == end_date_pt)[0][0] #457
    meteo_dates_pt = meteo_dates_pt[lower_cutoff_pt:upper_cutoff_pt]
    


    



    #read from file and filter dates
    meteo = np.genfromtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', usecols=(3,4,5,6,7,8,9), skip_header=1)
    meteo_pt = np.genfromtxt(base_path+'meteo/nhdhr_'+name+'_meteo.csv', delimiter=',', usecols=(3,4,5,6,7,8,9), skip_header=1)
    meteo = meteo[lower_cutoff:upper_cutoff,:]
    meteo_pt = meteo_pt[lower_cutoff_pt:upper_cutoff_pt,:]

    #normalize data
    meteo_norm = (meteo - mean_feats[:]) / std_feats[:]
    meteo_norm_pt = (meteo_pt - mean_feats[:]) / std_feats[:]

    ################################################################################
    # read/format GLM temperatures and observation data for numpy
    ###################################################################################
    n_total_dates = glm_temps.shape[0]
    n_total_dates_pt = glm_temps_pt.shape[0]


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

    # if n_dates != meteo.shape[0]:
    #     raise Exception("dates dont match")
    #     print(n_dates)
    #     print(meteo.shape[0])
    #     sys.exit()

    #data shape checks
    # assert n_dates == meteo.shape[0]
    # assert n_dates_pt == meteo_pt.shape[0]
    # assert n_dates == meteo_norm.shape[0]
    # assert n_dates_pt == meteo_norm_pt.shape[0]
    # assert n_dates == glm_temps.shape[0]
    # assert n_dates_pt == glm_temps_pt.shape[0]
    # assert(glm_temps[0,-1] == meteo_dates[0])
    # assert(glm_temps_pt[0,-1] == meteo_dates_pt[0])
    
    # if glm_temps[-1,-1] != meteo_dates[-1]:
        # print(glm_temps[-1,-1])
        # print(meteo_dates[-1])
        # raise Exception("dates dont match")
        # sys.exit()
    
    # if glm_temps_pt[-1,-1] != meteo_dates_pt[-1]:
        # print(glm_temps_pt[-1,-1])
        # print(meteo_dates_pt[-1])
        # raise Exception("dates don't match")
        # sys.exit()
    # assert(glm_temps[-1,-1] == meteo_dates[-1])
    # assert(glm_temps_pt[-1,-1] == meteo_dates_pt[-1])
    # glm_temps = glm_temps[:,0]
    # glm_temps_pt = glm_temps_pt[:,0]
    # obs = obs.values[:,1:] #remove needless nhd column
    # n_obs = obs.shape[0]

    # print("pretrain dates: ", start_date_pt, "->", end_date_pt)
    print("train dates: ", start_date, "->", end_date)
    ############################################################
    #fill numpy matrices
    ##################################################################

    #list static feats fo EA-LSTM
    static_feats = ['surface_area','SDF','K_d',\
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
    for feat in static_feats:
        pdb.set_trace()


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
    # if not os.path.exists("../../data/processed/"+name): 
        # os.mkdir("../../data/processed/"+name)
    # if not os.path.exists("../../models/"+name):
        # os.mkdir("../../models/"+name)
    # feat_path_pt = "../../data/processed/"+name+"/features_pt"
    feat_path = "../../data/processed/"+name+"/features_ea"
    # norm_feat_path_pt = "../../data/processed/"+name+"/processed_features_pt"
    norm_feat_path = "../../data/processed/"+name+"/processed_features_ea"
    # glm_path_pt = "../../data/processed/"+name+"/glm_pt"
    # glm_path = "../../data/processed/"+name+"/glm"
    # trn_path = "../../data/processed/"+name+"/train"
    # tst_path = "../../data/processed/"+name+"/test"
    # full_path = "../../data/processed/"+name+"/full"
    # dates_path = "../../data/processed/"+name+"/dates"
    # dates_path_pt = "../../data/processed/"+name+"/dates_pt"



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

print(no_obs_ids)
print(repr(n_obs_per))
pdb.set_trace()




