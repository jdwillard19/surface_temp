import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from joblib import dump, load
import re
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score

##################################################################3
# (Jan 2020 - Jared) - 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to


# metadata = pd.read_csv("../../metadata/conus_source_metadata.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")

# train_lakes = metadata['site_id'].values

#############################
#load data
# train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")



columns = ['Surface_Area','Latitude','Longitude', 
     'Elevation','ShortWave','LongWave','AirTemp','WindSpeedU','WindspeedV',
     # 'ShortWave_t-4','LongWave_t-4','AirTemp_t-4','WindSpeedU_t-4','WindSpeedV_t-4',
     # 'ShortWave_t-3','LongWave_t-3','AirTemp_t-3','WindSpeedU_t-3','WindSpeedV_t-3',
     # 'ShortWave_t-2','LongWave_t-2','AirTemp_t-2','WindSpeedU_t-2','WindSpeedV_t-2',\
     # 'ShortWave_t-1','LongWave_t-1','AirTemp_t-1','WindSpeedU_t-1','WindSpeedV_t-1',\
     # 'ShortWave_t-14','LongWave_t-14','AirTemp_t-14','WindSpeedU_t-14','WindSpeedV_t-14',\
     # 'ShortWave_t-30','LongWave_t-30','AirTemp_t-30','WindSpeedU_t-30','WindSpeedV_t-30',\
     'Surface_Temp']
# X = np.array(c)
# new_c = np.append(
#                   np.append(
#                             np.append(
#                                       X[i,:],
#                                       X[i-lookback:i,4:].flatten()),
#                             X[i-14,4:])
#                   ,X[i-30,4:])
# train_df = pd.DataFrame(columns=columns)
k = int(sys.argv[1])
param_search = True

# lookback = 4
# farthest_lookback = 30
#build training set
# k = int(sys.argv[1])
# save_file_path = '../../models/xgb_lagless_surface_temp_fold'+str(k)+"_03012021.joblib"

# final_output_df = pd.DataFrame()
# result_df = pd.DataFrame(columns=['site_id','temp_pred_xgb','temp_actual'])

train_lakes = metadata[metadata['5fold_fold']!=k]['site_id'].values

# lakenames = metadata['site_id'].values
# test_lakes = metadata[metadata['5fold_fold']==k]['site_id'].values
# assert(np.isin(train_lakes,test_lakes,invert=True).all())
train_df = pd.DataFrame(columns=columns)
# test_df = pd.DataFrame(columns=columns)

for ct, lake_id in enumerate(train_lakes):
    print(ct)
    if ct == 108:
      pdb.set_trace()
    # if ct %100 == 0:
    print(" assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea_conus_021621.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    inds = np.where(np.isfinite(y))[0]
    # inds = inds[np.where(inds > farthest_lookback)[0]]
    X = np.array([X[i,:] for i in inds],dtype = np.float)
    y = y[inds]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)

    # data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)

X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)
print("train set dimensions: ",X.shape)

if param_search:
    gbm = xgb.XGBRegressor(booster='gbtree')
    nfolds = 3
    parameters = {'objective':['reg:squarederror'],
                  'learning_rate': [.025, 0.05], #so called `eta` value
                  'max_depth': [6],
                  'min_child_weight': [11],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [5000,10000], #number of trees, change it to 1000 for better results
                  # 'n_estimators': [5000,10000,15000], #number of trees, change it to 1000 for better results
                  }
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


    parameters = gb_param_selection(X, y, nfolds)
    print(parameters)


