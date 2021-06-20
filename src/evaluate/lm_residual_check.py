import pandas as pd
import numpy as np
import pdb
import sys
import os
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
save_file_path = '../../models/lm_surface_temp_alldata.joblib'


# metadata = pd.read_csv("../../metadata/conus_source_metadata.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")

train_lakes = metadata['site_id'].values

#############################
#load data
# train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

columns = ['Surface_Area','Latitude','Longitude', 'Elevation',
           # 'ShortWave_t-30','LongWave_t-30','AirTemp_t-30','WindSpeedU_t-30','WindSpeedV_t-30',\
           # 'ShortWave_t-14','LongWave_t-14','AirTemp_t-14','WindSpeedU_t-14','WindSpeedV_t-14',\
           # 'ShortWave_t-4','LongWave_t-4','AirTemp_t-4','WindSpeedU_t-4','WindSpeedV_t-4',\
           # 'ShortWave_t-3','LongWave_t-3','AirTemp_t-3','WindSpeedU_t-3','WindSpeedV_t-3',\
           # 'ShortWave_t-2','LongWave_t-2','AirTemp_t-2','WindSpeedU_t-2','WindSpeedV_t-2',\
           # 'ShortWave_t-1','LongWave_t-1','AirTemp_t-1','WindSpeedU_t-1','WindSpeedV_t-1',\
           'ShortWave','LongWave','AirTemp','WindSpeedU','WindspeedV',\
           'Surface_Temp']

train_df = pd.DataFrame(columns=columns)

param_search = True
k = int(sys.argv[1])
# lookback = 4
# farthest_lookback = 30
#build training set

final_output_df = pd.DataFrame()
result_df = pd.DataFrame(columns=['site_id','temp_pred_lm','temp_actual'])

train_lakes = metadata['site_id'].values
# lakenames = metadata['site_id'].values
# test_lakes = metadata[metadata['5fold_fold']==k]['site_id'].values
train_df = pd.DataFrame(columns=columns)
# test_df = pd.DataFrame(columns=columns)

# for ct, lake_id in enumerate(train_lakes):
#     if ct %100 == 0:
#       print("fold ",k," assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
#     #load data
#     feats = np.load("../../data/processed/"+lake_id+"/features_ea_conus_021621.npy")
#     labs = np.load("../../data/processed/"+lake_id+"/full.npy")
#     # dates = np.load("../../data/processed/"+name+"/dates.npy")
#     data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
#     X = data[:,:-1]
#     y = data[:,-1]
#     inds = np.where(np.isfinite(y))[0]
#     if inds.shape[0] == 0:
#         continue
#     # inds = inds[np.where(inds > farthest_lookback)]

#     # if lookback > 0:
#         # X = np.array([np.append(np.append(np.append(X[i,:],X[i-lookback:i,4:].flatten()),X[i-14,4:]),X[i-30,4:]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
#     X = np.array([X[i,:] for i in inds],dtype = np.float)
#     # y = y[farthest_lookback:]
#     y = y[inds]
#     #remove days without obs
#     data = np.concatenate((X,y.reshape(len(y),1)),axis=1)

#     data = data[np.where(np.isfinite(data[:,-1]))]
#     new_df = pd.DataFrame(columns=columns,data=data)
#     train_df = pd.concat([train_df, new_df], ignore_index=True)
# X = train_df[columns[:-1]].values
# y = np.ravel(train_df[columns[-1]].values)

# np.save("./lm_train_x",X)
# np.save("./lm_train_y",y)

X = np.load("./lm_train_x.npy")
y = np.load("./lm_train_y.npy")
print("train set dimensions: ",X.shape)
#construct lookback feature set??
model = LinearRegression()

print("Training linear model...fold ",k)
model.fit(X, y)
dump(model, save_file_path)
print("model trained and saved to ", save_file_path)

#test
for ct, lake_id in enumerate(train_lakes):
    print("fold ",k," testing test lake ",ct,"/",len(train_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea_conus_021621.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    inds = np.where(np.isfinite(y))[0]
    if inds.shape[0] == 0:
        continue
        # X = np.array([np.append(np.append(np.append(X[i,:],X[i-lookback:i,4:].flatten()),X[i-14,4:]),X[i-30,4:]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
    X = np.array([X[i,:] for i in inds],dtype = np.float)
    # y = y[farthest_lookback:]
    y = y[inds]
    dates = dates[inds]

    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)
    data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    X = new_df[columns[:-1]].values
    y_act = np.ravel(new_df[columns[-1]].values)
    y_pred = model.predict(X)

    df = pd.DataFrame()
    df['temp_pred_lm'] = y_pred
    df['temp_actual'] = y_act
    df['Date'] = dates
    df['site_id'] = lake_id
    result_df = result_df.append(df)

#       # test_df = pd.concat([test_df, new_df], ignore_index=True)
result_df.reset_index(inplace=True)
result_df.to_feather("../../results/lm_lagless_061921.feather")
# if param_search:
#     gbm = xgb.XGBRegressor(booster='gbtree',n_estimators=10000,learning_rate=.025,max_depth=6,min_child_weight=11,subsample=.8,colsample_bytree=.7)
#     nfolds = 3
#     parameters = {'objective':['reg:squarederror'],
#                   'learning_rate': [.025, 0.05], #so called `eta` value
#                   'max_depth': [6],
#                   'min_child_weight': [11],
#                   'subsample': [0.8],
#                   'colsample_bytree': [0.7],
#                   'n_estimators': [5000,10000,15000], #number of trees, change it to 1000 for better results
#                   }
#     def gb_param_selection(X, y, nfolds):
#         # ests = np.arange(1000,6000,600)
#         # lrs = [.05,.01]
#         # max_d = [3, 5]
#         # param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
#         # grid_search = GridSearchCV(gbm, param_grid, cv=nfolds, n_jobs=-1,verbose=1)
#         grid_search = GridSearchCV(gbm, parameters, n_jobs=-1, cv=nfolds,verbose=1)
#         grid_search.fit(X, y)
#         # print(grid_search.best_params_)
#         return grid_search.best_params_


#     parameters = gb_param_selection(X, y, nfolds)
#     print(parameters)