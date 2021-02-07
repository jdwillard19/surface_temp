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
test_lakes = np.load("../../data/static/lists/target_lakes_wrr.npy",allow_pickle=True)




columns = ['ShortWave','LongWave','AirTemp','WindSpeed','Surface_Area','Surface_Temp']
feat_inds = [0,1,2,4,8]
train_df = pd.DataFrame(columns=columns)

lookback = 4
farthest_lookback = 30
#build training set
for site_ct, site_id in enumerate(test_lakes):
    # if site_ct < 58:
    #     continue
    print("predicting site ",site_ct,"/",len(test_lakes))
    #load data
    feats = np.load("../../data/processed/"+site_id+"/features_ea.npy")
    labs = np.load("../../data/processed/"+site_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,feat_inds],labs.reshape(labs.shape[0],1)),axis=1)

    X = data[:,:-1]
    y = data[:,-1]

    if lookback > 0:
        X = np.array([np.append(np.append(np.append(X[i,:],X[i-lookback:i,:4].flatten()),X[i-14,:4]),X[i-30,:4]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
        y = y[farthest_lookback:]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)
    #remove days without obs
    data = data[np.where(np.isfinite(data[:,-1]))]
    X = data[:,:-1]
    y = data[:,-1]
    y_pred = model.predict(X)
    y_act = y
    rmse = calc_rmse(y_pred,y_act)
    print("rmse: ", rmse)
    result_df = result_df.append(pd.DataFrame({'site_id': ['nhdhr_'+site_id], 'XGB_rmse': [rmse]}))

print("median rmse ", np.median(result_df['XGB_rmse']))
print("q1 rmse ", np.quantile(result_df['XGB_rmse'],.25))
print("q3 rmse ", np.quantile(result_df['XGB_rmse'],.75))
pdb.set_trace()