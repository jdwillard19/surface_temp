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
save_file_path = '../../models/xgb_surface_temp.joblib'




#############################
#load data
train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

columns = ['ShortWave_t-14','LongWave_t-14','AirTemp_t-14','WindSpeed_t-14',\
           'ShortWave_t-4','LongWave_t-4','AirTemp_t-4','WindSpeed_t-4',\
           'ShortWave_t-3','LongWave_t-3','AirTemp_t-3','WindSpeed_t-3',\
           'ShortWave_t-2','LongWave_t-2','AirTemp_t-2','WindSpeed_t-2',\
           'ShortWave_t-1','LongWave_t-1','AirTemp_t-1','WindSpeed_t-1',\
           'ShortWave','LongWave','AirTemp','WindSpeed',\
           'Surface_Area','Surface_Temp']
feat_inds = [0,1,2,4,8]
train_df = pd.DataFrame(columns=columns)

param_search = True

lookback = 4
farthest_lookback = 14
#build training set
for ct, lake_id in enumerate(train_lakes):
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,feat_inds],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    if lookback > 0:
        X = np.array([np.append(np.append(X[i,:],X[i-lookback:i,:4].flatten()),X[i-14,:4]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
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
if param_search:
    gbm = xgb.XGBRegressor(booster='gbtree')
    nfolds = 12
    parameters = {'objective':['reg:squarederror'],
                  'learning_rate': [.125,.025, 0.05], #so called `eta` value
                  'max_depth': [6],
                  'min_child_weight': [11],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [4000,7000,10000], #number of trees, change it to 1000 for better results
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

#no lookback params
# parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 2000, 'objective': 'reg:squarederror', 'subsample': 0.8}

#1 lookback params
#parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 4000, 'objective': 'reg:squarederror', 'subsample': 0.8}

#2 lookback params
#parmaeters = {'colsample_bytree': 0.7, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 6000, 'objective': 'reg:squarederror', 'subsample': 0.8}


#3 lookback params
# parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 8000, 'objective': 'reg:squarederror', 'subsample': 0.8}

#4 lookback params
# parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 8000, 'objective': 'reg:squarederror', 'subsample': 0.8}


#create and fit model
model = xgb.XGBRegressor(booster='gbtree', **parameters)

cv = cross_val_score(model, X, y=y, cv=12, n_jobs=12, verbose=1)
print("cv scores ", cv)
print(np.mean(cv))
# sys.exit()
pdb.set_trace()
print("Training XGB regression model...")
model.fit(X, y)
dump(model, save_file_path)
print("model trained and saved to ", save_file_path)





