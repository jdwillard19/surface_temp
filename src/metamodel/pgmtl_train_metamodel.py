import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from joblib import dump, load
import re
import datetime

##################################################################3
# (Sept 2020 - Jared) - PG-MTL training script on 145 source lake 
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 5500, etc)
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to
save_file_path = '../../models/metamodel_xgb_pgdl.joblib'
# save_file_path = '../../models/metamodel_pgdl_RMSE_GBR.joblib'

#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
# feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
#        'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
#        'perc_dif_sqrt_surface_area']

feats = ['n_obs_sp', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_mean_airdif',
       'dif_surface_area', 'dif_sw_mean', 'dif_sw_mean_au', 'dif_lw_std_au',
       'dif_at_std_au', 'dif_snow_mean_au', 'dif_zero_temp_doy',
       'perc_dif_surface_area']

###################################################################################


#######################################################################3
#paste hyperparameters found in "pbmtl_hyperparameter_search.py" here
#
#
n_estimators = 300
objective = 'reg:squarederror'
learning_rate = .05
colsample_bytree=.7
max_depth=6
min_child_weight=11
subsample=.8
#####################################################################


#{'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 300, 'objective': 'reg:squarederror', 'subsample': 0.8}

########################
##########################
#metamodel training code
##########################
#######################




train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]


#compile training data
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



#train model
X_trn = pd.DataFrame(train_df[feats])
y_trn = np.array([float(x) for x in np.ravel(pd.DataFrame(train_df['rmse']))])
# model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr)
# print("Model training in progress...")
model = xgb.XGBRegressor(booster='gbtree',n_estimators=n_estimators,objective=objective,learning_rate=learning_rate,colsample_bytree=colsample_bytree,
             max_depth=6,min_child_weight=min_child_weight,subsample=subsample)
# model = RandomForestRegressor(n_estimators=n_estimators)
print("Training metamodel...")
model.fit(X_trn, y_trn)
dump(model, save_file_path)
print("Training Complete, saved to ", save_file_path)
