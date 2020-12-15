import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../../data')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import re 
from sklearn.metrics import mean_squared_error
import xgboost as xgb

train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]
n_lakes = len(train_lakes)

#cv params
nfolds = 24


# Feats found in "pgmtl_feature_selection.py pasted here"
##################################################################################
# feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
#        'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
#        'perc_dif_sqrt_surface_area']

# feats = ['n_obs', 'n_obs_sp', 'n_obs_su', 'n_obs_au', 'obs_temp_mean',
#        'obs_temp_std', 'obs_temp_skew', 'obs_temp_kurt', 'ad_zero_temp_doy',
#        'ad_at_amp', 'ad_ws_sp_mix', 'obs_temp_mean_airdif', 'dif_SDF',
#        'dif_k_d', 'dif_lat', 'dif_long', 'dif_surface_area', 'dif_sw_mean',
#        'dif_sw_std', 'dif_lw_mean', 'dif_lw_std', 'dif_at_std', 'dif_rh_mean',
#        'dif_rh_std', 'dif_ws_mean', 'dif_ws_std', 'dif_rain_mean',
#        'dif_rain_std', 'dif_snow_std', 'dif_sw_mean_sp', 'dif_sw_std_sp',
#        'dif_lw_mean_sp', 'dif_lw_std_sp', 'dif_at_mean_sp', 'dif_at_std_sp',
#        'dif_rh_mean_sp', 'dif_rh_std_sp', 'dif_ws_mean_sp', 'dif_ws_std_sp',
#        'dif_rain_mean_sp', 'dif_rain_std_sp', 'dif_snow_std_sp',
#        'dif_sw_mean_su', 'dif_sw_std_su', 'dif_lw_mean_su', 'dif_lw_std_su',
#        'dif_at_mean_su', 'dif_at_std_su', 'dif_rh_mean_su', 'dif_rh_std_su',
#        'dif_ws_mean_su', 'dif_ws_std_su', 'dif_rain_mean_su',
#        'dif_rain_std_su', 'dif_snow_mean_su', 'dif_snow_std_su',
#        'dif_sw_mean_au', 'dif_sw_std_au', 'dif_lw_mean_au', 'dif_lw_std_au',
#        'dif_at_mean_au', 'dif_at_std_au', 'dif_rh_mean_au', 'dif_rh_std_au',
#        'dif_ws_mean_au', 'dif_ws_std_au', 'dif_rain_mean_au',
#        'dif_rain_std_au', 'dif_snow_std_au', 'dif_sw_mean_wi', 'dif_sw_std_wi',
#        'dif_lw_mean_wi', 'dif_lw_std_wi', 'dif_at_mean_wi', 'dif_at_std_wi',
#        'dif_rh_std_wi', 'dif_ws_mean_wi', 'dif_ws_std_wi', 'dif_rain_mean_wi',
#        'dif_rain_std_wi', 'dif_snow_mean_wi', 'dif_snow_std_wi',
#        'dif_zero_temp_doy', 'dif_at_amp', 'dif_ws_sp_mix',
#        'perc_dif_surface_area', 'dif_sqrt_surface_area',
#        'perc_dif_sqrt_surface_area']

#W TRANSFER OPTIM, W PRETRAIN
feats = ['n_obs_sp', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_mean_airdif',
       'dif_surface_area', 'dif_sw_mean', 'dif_sw_mean_au', 'dif_lw_std_au',
       'dif_at_std_au', 'dif_snow_mean_au', 'dif_zero_temp_doy',
       'perc_dif_surface_area']

#NO TRAN NO PRETRAIN
# feats = ['n_obs', 'n_obs_sp', 'n_obs_su', 'n_obs_au', 'obs_temp_mean',
#        'obs_temp_skew', 'obs_temp_kurt', 'obs_temp_mean_airdif',
#        'dif_surface_area', 'dif_lw_std', 'dif_at_std', 'dif_snow_mean',
#        'dif_rh_std_su', 'dif_snow_mean_su', 'dif_sw_mean_au', 'dif_lw_mean_au',
#        'dif_lw_std_au', 'dif_at_std_au', 'dif_rh_std_au', 'dif_rain_mean_au',
#        'dif_snow_mean_au', 'dif_lw_std_wi', 'dif_rain_mean_wi',
#        'perc_dif_surface_area']




# feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area',
#        'dif_rh_mean_au', 'dif_lathrop_strat', 'dif_glm_strat_perc',
#        'perc_dif_max_depth', 'perc_dif_surface_area',
#        'perc_dif_sqrt_surface_area']
####################################################################################


train_df = pd.DataFrame()


for _, lake_id in enumerate(train_lakes):
    new_df = pd.DataFrame()

   #get performance results (metatargets), filter out target as source
    lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results",header=None,names=['source_id','rmse'])
    lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

    #get metadata differences between target and all the sources
    lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
    lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
    lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
    new_df = lake_df
    train_df = pd.concat([train_df, new_df], ignore_index=True)


parameters = {'objective':['reg:squarederror'],
              'learning_rate': [0.05, .1], #so called `eta` value
              'max_depth': [6,8],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000,2000,4000], #number of trees, change it to 1000 for better results
              }
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))
gbm = xgb.XGBRegressor(booster='gbtree')

def gb_param_selection(X, y, nfolds):
    # ests = np.arange(1000,6000,600)
    # lrs = [.05,.01]
    # max_d = [3, 5]
    # param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
    # grid_search = GridSearchCV(gbm, param_grid, cv=nfolds, n_jobs=-1,verbose=1)
    grid_search = GridSearchCV(gbm, parameters, n_jobs=-1, cv=nfolds,verbose=1)
    grid_search.fit(X, y)
    # print(grid_search.best_params_)
    return grid_search.best_params_


print(gb_param_selection(X, y, nfolds))

