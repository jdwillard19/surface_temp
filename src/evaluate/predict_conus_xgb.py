import pandas as pd
import numpy as np
import pdb
import sys
import os
from joblib import dump, load
import re
import datetime
import xgboost as xgb

##################################################################3
# (Jan 2020 - Jared) - 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save results  to
save_file_path = '../../results/global_xgb.csv'
result_df = pd.DataFrame(columns=['site_id','XGB_rmse'])

def calc_rmse(predictions,targets):
    n = len(predictions)
    return np.linalg.norm(predictions - targets) / np.sqrt(n)
#############################
#load data

#create and fit model
model_path = '../../models/xgb_surface_temp.joblib'

model = load(model_path)
metadata = pd.read_csv("../../metadata/conus_source_metadata.csv")
train_lakes = metadata['site_id'].values

#############################
#load data
# train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

columns = ['Surface_Area','Latitude','Longitude',
           'ShortWave_t-30','LongWave_t-30','AirTemp_t-30','WindSpeedU_t-30','WindSpeedV_t-30',\
           'ShortWave_t-14','LongWave_t-14','AirTemp_t-14','WindSpeedU_t-14','WindSpeedV_t-14',\
           'ShortWave_t-4','LongWave_t-4','AirTemp_t-4','WindSpeedU_t-4','WindSpeedV_t-4',\
           'ShortWave_t-3','LongWave_t-3','AirTemp_t-3','WindSpeedU_t-3','WindSpeedV_t-3',\
           'ShortWave_t-2','LongWave_t-2','AirTemp_t-2','WindSpeedU_t-2','WindSpeedV_t-2',\
           'ShortWave_t-1','LongWave_t-1','AirTemp_t-1','WindSpeedU_t-1','WindSpeedV_t-1',\
           'ShortWave','LongWave','AirTemp','WindSpeedU','WindspeedV',\
           'Surface_Temp']

train_df = pd.DataFrame(columns=columns)

param_search = True

lookback = 4
farthest_lookback = 30
#build training set
for ct, lake_id in enumerate(train_lakes):
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea_conus.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    if lookback > 0:
        X = np.array([np.append(np.append(np.append(X[i,:],X[i-lookback:i,3:].flatten()),X[i-14,3:]),X[i-30,3:]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
        y = y[farthest_lookback:]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)

    data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)


X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)

print("train set dimensions: ",X.shape)
#construct lookback feature set??